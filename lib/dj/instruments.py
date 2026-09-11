"""Per-song INSTRUMENT discovery + the note each one plays per beat,
from a track's rendered stems.

Stems say WHICH KIND of material plays where (drums / bass / other /
vocals). This pass goes one level down: inside each stem it finds the
individual sounds the SONG is built from - not a fixed taxonomy, the
clusters of this recording - and reads what each of them does on every
beat of the DJ grid (16th-note steps inside the beat), pitched where the
sound has a pitch.

Two kinds of instrument per stem:

  hits      every onset in the stem gets a timbre vector and the onsets
            are clustered -> the stem's palette of struck / plucked
            sounds. Drums: attack + body mel spectra, decay, centroid;
            the clusters OWN their onsets, and their templates (mel
            frames at the onset, 23 and 58 ms later, stacked - spectrum
            AND envelope) are explained by each other with NNLS that may
            not overshoot, so a cluster that is only a coincidence of
            other sounds (a kick under a clap) credits its hits to each
            component and goes, one sound split by velocity merges, and
            a coincidence with something of its own keeps the residual.
            (Whole-stem NMF peak picking was tried first: it splits
            spectrally alike sounds - a hat, a shaker - arbitrarily.)
            Pitched stems: every onset gets a pitch by harmonic summation
            with everything already sounding subtracted (the spectrum
            just before the onset), and its timbre is the HARMONIC
            PROFILE (levels of the first twelve harmonics) + decay - the
            same pluck on different notes looks the same, so an
            instrument is one cluster whatever line it plays. Onsets
            with no pitch (percussion leaked into the stem, vocal chops)
            cluster on their mel timbre as unpitched instruments.
  sustain   the harmonic (HPSS) part of a pitched stem, beat by beat: a
            note counts only if it is prominent through BOTH halves of
            the beat (a pluck's tail is loud at the start and gone by
            the end; a pad holds). Beats cluster by spectral envelope
            (MFCC) into sustained instruments, each with a per-beat
            chord of up to POLY notes.

Every instrument carries a per-song description made from what was
measured (register, envelope, note range, hit count, level relative to
the loudest sound of the stem), an EXEMPLAR (start, end) in the stem the
UI can cut and play, a role HINT (kick? pad?) that is only a reading aid,
and its event list:

    [beat, step, midi | None, dur_steps, vel 0..1, conf 0..1, offset_s]

offset_s = where the onset really was, relative to the grid step (a
pushed or lazy hit keeps its feel when replayed; the step stays the
musical address).

plus level_dbfs, the level (dBFS, 60 ms rms) of the instrument's loud
hits - what vel is measured against, so a reconstruction can put the
exemplar back at the song's level (lib/dj/resynth.py).

beat = index into result["beats"] (the DJ grid's beats, main segment
extrapolated over the whole track), step = 16th inside the beat (0..3).

Storage: <music_root>/.stems/<track_id>/instruments.json next to the stem
files (the filesystem is the source of truth for stems, so their
derivatives live with them). INSTRUMENTS_VERSION gates re-analysis.

Pure numpy / librosa / scikit-learn - no torch. The onset detection,
onset-timbre vectors, the drum k-means and the exemplar picking are
shared with lib/gen/analysis/sounds.py (the generative console's sound
identification); imported lazily so this module stays light to import.
"""
import json
import os

import numpy as np

RATE = 44100
INSTRUMENTS_VERSION = 14              # v14: v13 + chord-tone level split, holds extended to where the stem lets go, each
                                      # YourMT3 voice's octave voted by the transcription reader, twins off, kick cap 4
                                      # v13: v12 + `other` from YourMT3+ (notes labelled by instrument, one voice per program,
                                      # duplicates across programs dropped), per-family drum cluster caps, parallel readers,
                                      # vocals carried as phrases only
                                      # v12: v11 + drums read per drumsep family (lib/dj/drumsep.py: kick / snare / toms / cymbals)
                                      # v11: v10 + bass odd-harmonic octave test, `other` continuation merge (within each voice)
                                      # + sub-octave ghost drop
                                      # v9: bass from a pitch track (lib/dj/bassreader.py), `other` from a transcription with voices by
                                      # timbre (lib/dj/polyreader.py), drum onsets at delta 0.03 and coincidences decomposed
                                      # (pure kicks no longer merge into kick+hat; multi-component credits) - judged on the
                                      # truth set (tools/tests/_dj_truthset.py): notes F1 bass 0.61 -> 0.80, other 0.29 -> 0.61
                                      # v8: held notes: overlapping same-pitch holds merged (the fusion duplicated them), the hold's own
                                      # level slope per voice (`hold_decay_db_s`), verified per-voice level steps (`gain_db`)
                                      # v7: every instrument checked against its stem by synthesis (lib/dj/explain.py): `explained`
                                      # per instrument, events in silence / voices that explain nothing / look-alike splits removed
                                      # when the rendered stem gets closer (result["pruned"], stems[..]["gap_db"])
                                      # v6: a coincidence's energy is split by ENERGY fractions incl. the residual; a mixed sound's level is its own part
FILE = "instruments.json"
STEPS = 4                              # 16th-note steps per beat
STEM_ORDER = ("drums", "bass", "other", "vocals")
PITCHED_STEMS = ("bass", "other", "vocals")
# MIDI search range per stem (bass 24 = C1 32.7 Hz; other up to C7).
PITCH_RANGE = {"bass": (24, 62), "other": (36, 96), "vocals": (43, 86)}
# spectrum window after an onset: long enough to resolve the register's semitones
POST_S = {"bass": 0.24, "other": 0.13, "vocals": 0.16}
MAX_HIT_INSTRUMENTS = {"drums": 9, "bass": 3, "other": 5, "vocals": 3}
MAX_SUSTAIN_INSTRUMENTS = {"bass": 1, "other": 3, "vocals": 2}
POLY = {"bass": 1, "other": 3, "vocals": 2}     # simultaneous sustained notes per beat
HIT_PITCH_CONF = 0.22                  # a plucked onset needs this to count as a note
HIT_FLOOR_DB = 18.0                    # a pitched stem's onset this far under its instrument's loud hits is bleed or
                                       # noise, not a note (measured: a quarter of bass "notes" sat there, mostly at
                                       # the range floor = kick bleed)
PITCHED_SHARE = 0.3                    # a cluster is a pitched instrument when this share of its hits has a pitch
SUSTAIN_FLOOR_DB = -30.0               # a beat's harmonic level under (95th pct - this) is silence
SUSTAIN_HOLD = 0.35                    # rms(last 40%) / rms(first 40%) below this = a decaying tail, not a hold
SUSTAIN_PROMINENT = 0.3                # a held note's salience in BOTH halves vs the louder half's best
SUSTAIN_TONAL = 4.0                    # peak / median salience across the semitone grid: below this the "held
                                       # note" is reverb, room or percussion residue (measured: a pad 15-23, a
                                       # pad-heavy song 5-16, a percussion tool 2-8)
SILHOUETTE_MIN = 0.35                  # below this a set of held beats is one instrument
SILHOUETTE_HITS = 0.4                  # pitched hits (raw dB features)
PROFILE_SEP = 6.0                      # dB rms between harmonic-profile centroids for two pitched instruments
_TEMPLATE_OFFSETS = (0, 4, 10)         # mel frames (0 / 23 / 58 ms) stacked into a drum template
_N_HPS = 8192
_HPSS_KERNEL = 17                      # librosa default 31: the median filter is the pass's cost

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi):
    if midi is None:
        return "-"
    m = int(midi)
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def path_for(music_root, track_id):
    from lib.dj.stems import stems_dir
    return os.path.join(stems_dir(music_root, track_id), FILE)


def load(music_root, track_id, any_version=False):
    """The stored result, or None (missing, unreadable, or an older
    INSTRUMENTS_VERSION unless any_version)."""
    p = path_for(music_root, track_id)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not any_version and d.get("version") != INSTRUMENTS_VERSION:
        return None
    return d


def save(music_root, track_id, result):
    p = path_for(music_root, track_id)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result, f, separators=(",", ":"))
    os.replace(tmp, p)
    return p


def has_instruments(music_root, track_id):
    """A CURRENT result exists (cheap: the version is the first key the
    writer emits, so the file's head decides - the library coverage line
    asks this for every track)."""
    p = path_for(music_root, track_id)
    try:
        with open(p, "rb") as f:
            head = f.read(48)
    except OSError:
        return False
    return f'"version":{INSTRUMENTS_VERSION},'.encode() in head


# --------------------------------------------------------------------------
# The beat grid, as the decks count it
# --------------------------------------------------------------------------

def beat_times(grid, downbeat_offset, duration_s, bpm=None):
    """DJ beat grid -> (beat times over the whole track, index of the
    first downbeat). The MAIN segment (longest) is extrapolated across
    the track: the mixer plays against that grid, so the notes are
    indexed the way the decks count. Falls back to a straight grid from
    bpm when there is no segment."""
    if grid:
        g = max(grid, key=lambda s: float(s["end_s"]) - float(s["start_s"]))
        period, first = float(g["period_s"]), float(g["first_beat_s"])
    elif bpm:
        period, first = 60.0 / float(bpm), 0.0
    else:
        return np.zeros(0), 0
    if period <= 0.05 or duration_s <= period:
        return np.zeros(0), 0
    n_before = int(np.floor(first / period + 1e-6))          # whole beats between t=0 and the first
    t0 = first - n_before * period
    n = int(np.floor((duration_s - t0) / period)) + 1
    times = t0 + np.arange(n) * period
    off = int(downbeat_offset or 0)
    # beat index i is (i - n_before) beats after the grid's first beat
    down0 = (off + n_before) % 4
    return times, down0


def beat_step_of(beats, t, period_hint=None):
    """(beat index, 16th step) of time t, or None outside the grid."""
    n = len(beats)
    if n == 0:
        return None
    k = int(np.searchsorted(beats, t, side="right") - 1)
    if k < 0:
        return None
    nxt = beats[k + 1] if k + 1 < n else beats[k] + (period_hint or (beats[k] - beats[k - 1] if k else 0.5))
    frac = (t - beats[k]) / max(nxt - beats[k], 1e-6)
    step = int(round(frac * STEPS))
    if step >= STEPS:
        k, step = k + 1, 0
        if k >= n:
            return None
    return k, step


def event_time(beats, beat, step=0):
    n = len(beats)
    if beat >= n:
        return float(beats[-1]) if n else 0.0
    nxt = beats[beat + 1] if beat + 1 < n else beats[beat] + (beats[beat] - beats[beat - 1] if beat else 0.5)
    return float(beats[beat] + step / STEPS * (nxt - beats[beat]))


# --------------------------------------------------------------------------
# Descriptions (per song, from measurements)
# --------------------------------------------------------------------------

def _register(centroid_hz):
    c = float(centroid_hz)
    return "sub" if c < 120 else "low" if c < 300 else "mid" if c < 1200 else "bright" if c < 4000 else "air"


def _envelope(dur_s):
    d = float(dur_s)
    if d < 0.08:
        return "click"
    if d < 0.2:
        return "short"
    if d < 0.5:
        return "medium"
    return "long"


def describe(inst):
    """Label + detail strings from the instrument's measured facts."""
    reg = _register(inst.get("centroid_hz", 1000.0))
    if inst["kind"] == "sustain":
        label = f"{reg} sustained"
    else:
        label = f"{reg} {_envelope(inst.get('dur_s', 0.2))}"
        if inst.get("pitched"):
            label += " pitched"
    parts = []
    if inst.get("range"):
        lo, hi = inst["range"]
        parts.append(f"{note_name(lo)}-{note_name(hi)}" if lo != hi else note_name(lo))
    parts.append(f"{inst.get('centroid_hz', 0):.0f} Hz")
    if inst["kind"] != "sustain":
        parts.append(f"{1000 * inst.get('dur_s', 0):.0f} ms")
    parts.append(f"{inst.get('n', 0)} {'beats' if inst['kind'] == 'sustain' else 'hits'}")
    if inst.get("level_db") is not None:
        parts.append(f"{inst['level_db']:+.0f} dB")
    return label, " · ".join(parts)


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------

def _mono(arr):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a.astype(np.float32).mean(axis=1)
    return np.ascontiguousarray(a, dtype=np.float32)


def _hpss(y):
    import librosa
    return librosa.effects.hpss(y, margin=(1.0, 3.0), kernel_size=_HPSS_KERNEL)


def _rms_env(y, win_s=0.01):
    w = max(8, int(win_s * RATE))
    m = len(y) // w
    if m < 2:
        return np.zeros(2, dtype=np.float32), w
    env = np.sqrt((y[: m * w].reshape(m, w).astype(np.float64) ** 2).mean(axis=1))
    return env.astype(np.float32), w


def _hit_duration_s(env, hop, s, next_s, drop_db=24.0, cap_s=2.0):
    """Seconds until the level after the hit at sample s falls drop_db
    under its peak (or the next hit / the cap)."""
    i0 = int(s // hop)
    i1 = int(min(next_s, s + cap_s * RATE) // hop)
    if i1 <= i0 + 1 or i0 >= len(env):
        return max(0.03, (next_s - s) / RATE) if next_s > s else 0.05
    seg = env[i0:min(i1, len(env))]
    pk = float(seg[: max(1, int(0.05 * RATE / hop))].max()) + 1e-9
    below = np.where(seg < pk * 10 ** (-drop_db / 20.0))[0]
    n = int(below[0]) if len(below) else len(seg)
    return max(0.03, n * hop / RATE)


def _fold_hits(beats, items, period):
    """[(sample, midi|None, dur_s, vel, conf)] -> events [beat, step, midi, dur_steps, vel, conf],
    one per (beat, step) (the strongest wins)."""
    out = {}
    step_s = period / STEPS
    for s, midi, dur_s, vel, conf in items:
        pos = beat_step_of(beats, s / RATE, period)
        if pos is None:
            continue
        dur = max(1, int(round(dur_s / max(step_s, 1e-6))))
        key = (pos[0], pos[1], (int(midi) if midi is not None else None))
        if key not in out or vel > out[key][4]:
            off = s / RATE - event_time(beats, pos[0], pos[1])
            out[key] = [int(pos[0]), int(pos[1]), (int(midi) if midi is not None else None), int(dur), round(float(vel), 3),
                        round(float(conf), 3), round(float(off), 4)]
    return [out[k] for k in sorted(out, key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2]))]


def _level_rel(levels):
    top = max(levels) if levels else 0.0
    return [round(float(v - top), 1) for v in levels]


def _cluster_auto(X, k_max, min_share=0.03, min_sil=SILHOUETTE_MIN, standardize=True, min_sep=0.0):
    """k-means with the number of clusters chosen by silhouette (k = 1
    when no split is convincing); clusters whose centroids sit closer
    than min_sep (RMS over features, in the features' own units - meant
    for dB features left unstandardised) are one cluster; clusters under
    min_share of the points fold into the nearest kept one. -> labels."""
    n = len(X)
    if n < 12 or k_max < 2:
        return np.zeros(n, dtype=int)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6) if standardize else np.asarray(X, dtype=np.float64)
    best, best_sil = None, min_sil
    rng = np.random.default_rng(0)
    sample = rng.choice(n, size=min(n, 1500), replace=False) if n > 1500 else None
    for k in range(2, int(min(k_max, n // 8)) + 1):
        lab = KMeans(k, n_init=4, random_state=0).fit(Xn).labels_
        if len(set(lab.tolist())) < 2:
            continue
        sil = silhouette_score(Xn[sample], lab[sample]) if sample is not None else silhouette_score(Xn, lab)
        if sil > best_sil:
            best, best_sil = lab, sil
    if best is None:
        return np.zeros(n, dtype=int)
    labels = best.copy()
    while min_sep > 0:
        ids = sorted(set(labels.tolist()))
        if len(ids) < 2:
            break
        cents = {i: Xn[labels == i].mean(axis=0) for i in ids}
        pair = None
        for a in ids:
            for b in ids:
                if b <= a:
                    continue
                d = float(np.sqrt(np.mean((cents[a] - cents[b]) ** 2)))
                if d < min_sep and (pair is None or d < pair[0]):
                    pair = (d, a, b)
        if pair is None:
            break
        labels[labels == pair[2]] = pair[1]
    ids = sorted(set(labels.tolist()))
    small = [i for i in ids if (labels == i).sum() < max(4, min_share * n)]
    keep = [i for i in ids if i not in small]
    if small and keep:
        cents = {i: Xn[labels == i].mean(axis=0) for i in keep}
        for j in np.where(np.isin(labels, small))[0]:
            labels[j] = min(keep, key=lambda i: float(np.linalg.norm(Xn[j] - cents[i])))
    return labels


# --------------------------------------------------------------------------
# Pitch: harmonic summation on a spectrum
# --------------------------------------------------------------------------

def _spectrum(seg, n_fft=None):
    n = len(seg)
    n_fft = n_fft or max(_N_HPS, 1 << (n - 1).bit_length())
    spec = np.abs(np.fft.rfft(seg.astype(np.float64) * np.hanning(n), n_fft)) / n
    return spec, RATE / float(n_fft)


def _harmonic_salience(R, df, midi_lo, midi_hi, n_h=6):
    """Salience per MIDI note (midi_lo..midi_hi) of a magnitude spectrum
    R with bin width df: sum over harmonics of the local peak level,
    weighted 1/h^0.6. Vectorised (a max filter replaces the peak search)."""
    from scipy.ndimage import maximum_filter1d
    Rm = maximum_filter1d(R, 5)
    midis = np.arange(midi_lo, midi_hi + 1)
    f0 = 440.0 * 2 ** ((midis - 69) / 12.0)
    sal = np.zeros(len(midis))
    for h in range(1, n_h + 1):
        idx = np.round(h * f0 / df).astype(int)
        ok = idx < len(Rm) - 3
        sal[ok] += Rm[idx[ok]] / (h ** 0.6)
    return sal, Rm


def _onset_pitch(y, s, midi_lo, midi_hi, post_s=0.13, pre_s=0.2):
    """Pitch of the sound that STARTS at sample s: the spectrum after the
    onset minus the spectrum before it (whatever was already sounding),
    then harmonic summation over the semitone grid.
    -> (midi | None, confidence 0..1, harmonic profile[12] in dB rel. max | None)."""
    a0, a1 = max(0, s - int(pre_s * RATE)), max(0, s - int(0.005 * RATE))
    b0, b1 = s + int(0.005 * RATE), min(len(y), s + int(post_s * RATE))
    if b1 - b0 < 1024 or a1 - a0 < 1024:
        return None, 0.0, None
    from scipy.ndimage import maximum_filter1d
    Pre, _ = _spectrum(y[a0:a1], _N_HPS)
    Post, df = _spectrum(y[b0:b1], _N_HPS)
    R = np.maximum(Post - 1.2 * Pre, 0.0)
    lo_i, hi_i = int(50.0 / df), int(6000.0 / df)
    total = float(R[lo_i:hi_i].sum()) + 1e-12
    if total < 1e-7:
        return None, 0.0, None
    sal, Rm = _harmonic_salience(R, df, midi_lo, midi_hi)
    if sal.max() <= 0:
        return None, 0.0, None
    best = midi_lo + int(np.argmax(sal))
    best_sal = float(sal.max())
    f0 = 440.0 * 2 ** ((best - 69) / 12.0)

    def pk(f):
        i = int(round(f / df))
        return float(Rm[i]) if i < len(Rm) else 0.0
    odd = pk(f0) + pk(3 * f0) + pk(5 * f0)
    even = pk(2 * f0) + pk(4 * f0) + pk(6 * f0)
    if odd < 0.3 * even and best + 12 <= midi_hi:
        best += 12
        f0 *= 2.0
    conf = min(1.0, best_sal / total * 3.0)
    # the runners-up, for the continuity pass (a line rarely leaps an octave for one note)
    order = np.argsort(sal)[::-1][:4]
    cands = [(midi_lo + int(i), float(sal[i] / best_sal)) for i in order if sal[i] > 0]
    if best not in [m for m, _s in cands]:
        cands.insert(0, (best, 1.0))
    # the profile: harmonic levels after the onset with what was already sounding subtracted, but
    # never more than 14 dB below the raw level - a harmonic the previous note shared would
    # otherwise read as a notch and split one instrument into "after E" and "after A"
    Postm = maximum_filter1d(Post, 5)
    prof = np.array([max(pk(h * f0), 0.2 * float(Postm[int(round(h * f0 / df))]) if int(round(h * f0 / df)) < len(Postm) else 0.0)
                     for h in range(1, 13)])
    prof = 20.0 * np.log10(prof + 1e-9)
    prof = np.clip(prof - prof.max(), -60.0, 0.0)
    return best, conf, prof, cands


def _viterbi_pitches(seq, lam=0.35, leap=19):
    """seq: [(t_s, [(midi, salience 0..1), ...])] in time order -> [midi]:
    the smoothest path through each onset's candidates - emission
    log(salience), transition cost lam * (|interval|/12 + 1 beyond a
    leap). Octave ghosts and bleed lose to a note that continues the
    line; a real leap still wins when its salience is clear."""
    n = len(seq)
    if n == 0:
        return []
    K = max(len(c) for _t, c in seq)
    score = np.full((n, K), -1e9)
    back = np.zeros((n, K), dtype=int)
    for k, (m, s) in enumerate(seq[0][1]):
        score[0, k] = np.log(s + 0.05)
    for i in range(1, n):
        prev = seq[i - 1][1]
        for k, (m, s) in enumerate(seq[i][1]):
            best, arg = -1e9, 0
            for j, (pm, _ps) in enumerate(prev):
                d = abs(m - pm)
                cost = lam * (min(d, 12) / 12.0 + (1.0 if d > leap else 0.0))
                v = score[i - 1, j] - cost
                if v > best:
                    best, arg = v, j
            score[i, k] = best + np.log(s + 0.05)
            back[i, k] = arg
    path = [int(np.argmax(score[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i, path[-1]]))
    path.reverse()
    return [seq[i][1][k][0] for i, k in enumerate(path)]


CLEAN_BASS_MIN_GAIN = 0.2      # below this correlation gain the kick is not in the bass stem; leave it alone


def clean_bass(bass, drums, fmax=180.0):
    """The bass stem with the drum stem's low band subtracted where they
    correlate: demucs leaves the kick in both, and the kick read as a
    note at the range floor was a quarter of all bass "notes"."""
    n = min(len(bass), len(drums))
    b, d = bass[:n].astype(np.float32), drums[:n].astype(np.float32)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, fmax / (RATE / 2), btype="low", output="sos")
    d_lo = sosfiltfilt(sos, d).astype(np.float32)
    b_lo = sosfiltfilt(sos, b).astype(np.float32)
    g = float(np.dot(b_lo, d_lo) / max(float(np.dot(d_lo, d_lo)), 1e-9))
    g = float(np.clip(g, 0.0, 1.5))
    if g < CLEAN_BASS_MIN_GAIN:
        # no bleed worth the name: subtracting a phase-misaligned sliver of the kick only ADDS a low
        # image (the synthetic gate's clean bass read eight F#1 notes at every kick after a 0.07 subtraction)
        return b.copy(), 0.0
    out = b.copy()
    out[:n] -= g * d_lo
    return out, g


_HARMONIC_INTERVALS = {12, 19, 24, 28, 31, 34, 36}     # semitones above a fundamental: harmonics 2, 3, 4, 5, 6, 7, 8


def _pick_notes(sal, midi_lo, poly, rel=0.45):
    """Top `poly` salience peaks that are not each other's octave ghost."""
    if sal is None or sal.max() <= 0:
        return []
    order = np.argsort(sal)[::-1]
    best = float(sal[order[0]])
    chosen = []
    for i in order:
        v = float(sal[i])
        if v < rel * best or len(chosen) >= poly:
            break
        m = midi_lo + int(i)
        if any(abs(m - c) <= 1 for c, _ in chosen):
            continue
        # a note sitting on a chosen note's harmonic series (octaves, the twelfth, two octaves + a
        # third...) is that note's harmonics unless it is nearly as loud
        if any((m - c) in _HARMONIC_INTERVALS and v < 0.8 * cv for c, cv in chosen):
            continue
        # local peak only (a semitone neighbour that is louder owns this energy)
        if (i > 0 and sal[i - 1] > v) or (i + 1 < len(sal) and sal[i + 1] > v):
            continue
        chosen.append((m, v))
    return [(m, v / best) for m, v in chosen]


# --------------------------------------------------------------------------
# DRUM instruments: onset clusters own their hits; coincidences are shared
# --------------------------------------------------------------------------

def _blocks(w, n_mels):
    """Energy of each stacked time block relative to the first."""
    b = np.array([float(w[i * n_mels:(i + 1) * n_mels].sum()) for i in range(len(_TEMPLATE_OFFSETS))])
    return b / max(b[0], 1e-12)


def _explain(W, i, others, r_neg_max=0.1):
    """NNLS (sqrt domain) of template i on `others`, pruning any
    component whose scaled template overshoots the target (a superset
    can never be a component of its own subset) until the fit no longer
    overshoots. -> (components, coef, r_pos, r_neg, shares)."""
    from scipy.optimize import nnls
    t = np.sqrt(W[:, i])
    ots = list(others)
    while ots:
        A = np.sqrt(W[:, ots])
        coef, _r = nnls(A, t)
        res = t - A @ coef
        tt = max(float((t ** 2).sum()), 1e-12)
        r_pos = float((np.maximum(res, 0.0) ** 2).sum() / tt)
        r_neg = float((np.maximum(-res, 0.0) ** 2).sum() / tt)
        if r_neg < r_neg_max or len(ots) == 1:
            contrib = coef * np.linalg.norm(A, axis=0)
            shares = contrib / max(float(contrib.sum()), 1e-12)
            # each component's fraction of the TARGET's ENERGY: the fit is in the sqrt-magnitude domain,
            # so a component's magnitude is coef^2 * W_j and energy is magnitude squared - a hat under a
            # kick is a large share of the sqrt-domain fit and a tiny share of the energy
            Wi = W[:, i]
            e_target = float(np.sum(Wi ** 2)) + 1e-12
            fracs = np.array([min(1.0, float(np.sum((coef[q] ** 2 * W[:, ots[q]]) ** 2)) / e_target) for q in range(len(ots))])
            return ots, coef, r_pos, r_neg, shares, fracs
        excess = [float((np.maximum(A[:, q] * coef[q] - t, 0.0) ** 2).sum()) for q in range(len(ots))]
        ots.pop(int(np.argmax(excess)))
    return [], np.zeros(0), 1.0, 0.0, np.zeros(0), np.zeros(0)


CREDIT_SHARE = 0.25            # a component of a coincidence is credited a hit from this sqrt-domain share up (see _resolve_clusters)
ONSET_CREDITS = True           # per-onset NNLS against the alive sounds: credits a sound under another's onset (see _onset_credits)
CREDIT_INSIDE = True           # a component at its usual level inside a sound that is mostly its own is credited (see _resolve_clusters)
CREDIT_SHARE_MULTI = 0.12      # in a coincidence of several sounds, a component with this sqrt-domain share gets the hits
MIN_COMPONENT_ONSETS = 6       # ...if it is a sound with at least this many onsets of its own
MERGE_MIN_COEF = 0.8           # a cluster merges into one other sound only when fitted at this share of its level or more
ONSET_CREDIT_LEVEL = 0.6       # ...when its coefficient puts it at this share of its own template level (sqrt domain)...
ONSET_CREDIT_FRAC = 0.04       # ...and it carries at least this fraction of the onset's energy


def _onset_credits(Vs, starts, fr, T, alive, W, own, credits, n_mels):
    """For every onset owned by an alive sound, NNLS of its patch on ALL
    alive templates; another sound fitted at >= ONSET_CREDIT_LEVEL of its
    template with >= ONSET_CREDIT_FRAC of the onset's energy is credited a
    hit there (onset, energy fraction, source cluster = the owner)."""
    from scipy.optimize import nnls
    A = np.sqrt(W[:, alive])
    norms = np.linalg.norm(A, axis=0) + 1e-12
    owner_of = {}
    for k in alive:
        for i in own[k]:
            owner_of[i] = k
    already = {k: set(o for o, _f, _s in credits[k]) for k in alive}
    for i, k_own in owner_of.items():
        f0 = fr(starts[i])
        if f0 + 2 > T:
            continue
        patch = Vs[:, max(0, f0 - 1):f0 + 2].mean(axis=1)
        t = np.sqrt(patch)
        coef, _r = nnls(A, t)
        e_target = float(np.sum(patch ** 2)) + 1e-12
        for q, k in enumerate(alive):
            if k == k_own or coef[q] < ONSET_CREDIT_LEVEL or i in already[k]:
                continue
            frac = min(1.0, float(np.sum((coef[q] ** 2 * W[:, k]) ** 2)) / e_target)
            if frac >= ONSET_CREDIT_FRAC:
                credits[k].append((i, frac, k_own))
                already[k].add(i)
    return credits


def _resolve_clusters(W, own, template, n_mels, same=0.12, mixed=0.6, top_share=0.7, credit_share=None, credit_level=0.6,
                      component_size=0.3, alike=0.97, env_tol=0.8):
    """W (dims, K) stacked templates, own {k: [onset idx]} per cluster,
    template(idx) -> a template for a set of onsets.

    First, clusters whose templates are all but identical (cosine >=
    alike, sqrt domain - one sound split by velocity or by what rang
    under it) merge. Then, quietest first, each template is explained
    by the other alive templates (_explain: NNLS that may not overshoot)
    and read by the unexplained share r_pos, the components' shares and
    whether the explanation's ENVELOPE (the stacked blocks) matches the
    target's - two sounds with one spectrum and different decays (a hat,
    a shaker) explain each other's attack, never each other's envelope:
      overshoot or envelope mismatch    -> a sound of its own
      r_pos < same, one component       -> the same sound: merge into it
      r_pos < same, several components  -> a pure coincidence (a kick
                                           under a hat): its onsets are
                                           CREDITED to each component with
                                           a real share, the cluster goes
      r_pos < mixed                     -> a coincidence with something of
                                           its own: keeps the residual as
                                           its template AND credits the
                                           components
      else                              -> a sound of its own
    A component counts only when its coefficient says it is there at
    near its usual level (credit_level: a kick's tail under a shaker, or
    a snare's low body read as half a kick, fit with a small coefficient
    and are not kick hits) AND it is an established sound (its own
    cluster at least component_size of this one: an 18-hit variant does
    not explain a 137-hit sound as a coincidence, it merges into it).
    A cluster with no such component is a sound of its own.
    Returns (alive ids, W, own, credits {k: [(onset idx, energy fraction)]}, own_frac {k: the
    fraction of its onsets' energy that is the sound's own - 1 for a pure sound, the residual's
    share for a sound that only plays under others})."""
    if credit_share is None:
        credit_share = CREDIT_SHARE
    W = np.array(W, dtype=np.float64)
    own = {k: list(v) for k, v in own.items()}
    credits = {k: [] for k in own}
    own_frac = {k: 1.0 for k in own}
    alive = list(own)

    def merge(i, j):
        own[j] += own[i]
        credits[j] += credits[i]                 # (onset, energy fraction, source cluster)
        alive.remove(i)
        W[:, j] = template(own[j])
    changed = True
    while changed and len(alive) > 1:
        changed = False
        S = np.sqrt(W[:, alive])
        S = S / (np.linalg.norm(S, axis=0) + 1e-12)
        C = S.T @ S
        best = None
        for a_ in range(len(alive)):
            for b_ in range(a_ + 1, len(alive)):
                if C[a_, b_] >= alike and (best is None or C[a_, b_] > best[0]):
                    best = (C[a_, b_], alive[a_], alive[b_])
        if best is not None:
            _c, i, j = best
            if len(own[i]) > len(own[j]):
                i, j = j, i
            merge(i, j)
            changed = True
    decided = set()
    for _ in range(2):
        for i in sorted(alive, key=lambda k: float(W[:, k].sum())):
            if i not in alive or i in decided:
                continue
            others = [j for j in alive if j != i]
            if not others:
                break
            comps, coef, r_pos, r_neg, shares, fracs = _explain(W, i, others)
            # components fitted under credit_level are BACKGROUND (a tail still ringing under this
            # sound): they may explain energy but they are not part of what this sound is
            cands = [q for q in range(len(comps)) if coef[q] >= credit_level]
            if len(cands) >= 2:
                # two or more sounds each at its usual level: a coincidence, whatever their cluster sizes (a
                # kick that always carries a hat has 8 pure-kick onsets against 107 kick+hat ones; the size
                # rule below is for the one-component "variant of the same sound" case)
                real = [q for q in cands if len(own[comps[q]]) >= MIN_COMPONENT_ONSETS]
            else:
                real = [q for q in cands if len(own[comps[q]]) >= component_size * len(own[i])]
            if not comps or not real or r_neg >= 0.1:
                decided.add(i)
                continue
            if len(real) < len(comps):
                # the decision must rest on the components that COUNT: a hat+snare cluster was explained by
                # the hat (0.92) plus a second hat+snare cluster at 0.47 (r_pos 0.04); dropping the latter as
                # "background" and merging into the hat on that r_pos lost the snare
                keep_ids = [comps[q] for q in real]
                comps, coef, r_pos, r_neg, shares, fracs = _explain(W, i, keep_ids)
                real = [q for q in range(len(comps)) if coef[q] >= credit_level]
                if not comps or not real or r_neg >= 0.1:
                    decided.add(i)
                    continue
            if r_pos >= mixed:
                # mostly a sound of its own, but a component sits inside it at its usual level without
                # overshoot: credit that component and keep the residual as the sound. A kick that always
                # has a hat on it never forms a pure cluster - its template IS kick+hat (truth set: 107 of
                # 130 kick onsets), and without this the hat on every kick beat was never credited
                if CREDIT_INSIDE:
                    rs_all = shares[real] / max(float(shares[real].sum()), 1e-12)
                    for q, sh in zip(real, rs_all):
                        if sh >= credit_share:
                            credits[comps[q]] += [(o, float(fracs[q]), i) for o in own[i]]
                    res = np.sqrt(W[:, i]) - np.sqrt(W[:, comps]) @ coef
                    res_lin = np.maximum(res, 0.0) ** 2
                    own_frac[i] = float(min(1.0, max(0.02, np.sum(res_lin ** 2) / (np.sum(W[:, i] ** 2) + 1e-12))))
                    W[:, i] = res_lin + 1e-9
                decided.add(i)
                continue
            fit = (np.sqrt(W[:, comps]) @ coef) ** 2
            env_ok = bool(np.abs(np.log(_blocks(fit, n_mels)[1:] + 1e-6) - np.log(_blocks(W[:, i], n_mels)[1:] + 1e-6)).max() <= env_tol)
            rs = shares[real] / max(float(shares[real].sum()), 1e-12)
            top = real[int(np.argmax(rs))]
            if r_pos < same:
                if not env_ok:
                    decided.add(i)
                    continue
                if len(real) == 1 and rs.max() >= top_share:
                    if coef[top] < MERGE_MIN_COEF:
                        # explained by ONE sound at well under its level: this is a SUBSET of that sound (the
                        # 8 pure kicks against the kick+hat cluster, coef 0.63), not a variant of it - keep
                        # it, so the mixture can be decomposed with it afterwards
                        decided.add(i)
                        continue
                    merge(i, comps[top])
                    continue
                # a pure coincidence of several sounds: every real component gets the hits (the hat under
                # the kick is a quarter of the sqrt-domain fit and was lost to the 0.25 threshold; the merge
                # into the top component used to take every onset for the kick and credit nobody)
                for q, sh in zip(real, rs):
                    if sh >= CREDIT_SHARE_MULTI:
                        credits[comps[q]] += [(o, float(fracs[q]), i) for o in own[i]]
                alive.remove(i)
                continue
            # a coincidence with something of its own: the residual is the sound; the components
            # get their FRACTION of the onset's energy (a kick under a snare is not a snare-loud kick)
            # and the sound itself keeps the residual's fraction
            for q, sh in zip(real, rs):
                if sh >= credit_share:
                    credits[comps[q]] += [(o, float(fracs[q]), i) for o in own[i]]
            res = np.sqrt(W[:, i]) - np.sqrt(W[:, comps]) @ coef
            res_lin = np.maximum(res, 0.0) ** 2
            own_frac[i] = float(min(1.0, max(0.02, np.sum(res_lin ** 2) / (np.sum(W[:, i] ** 2) + 1e-12))))
            W[:, i] = res_lin + 1e-9
            decided.add(i)
    return alive, W, own, credits, own_frac


RARE_SOUNDS = False            # after the clusters are resolved, onsets the alive sounds leave mostly unexplained are
                               # re-clustered into sounds of their own (_rare_sounds): toms, crashes, a rim - 5-10 hits a
                               # song, under the clusterer's minimum share, so k-means folded them into the nearest big sound.
                               # OFF: measured 2026-09-09 on the truth set - it finds nothing, because 4-5 broadband templates
                               # with free NNLS gains explain ANY onset patch to within 7 % (tom and crash onsets: r_pos
                               # 0.01-0.07, the same as pure hats); the residual is not where a rare sound shows
RARE_UNEXPLAINED = 0.5         # the share of an onset's patch energy the alive templates leave (positive residual, sqrt domain)
RARE_MIN = 4                   # a rare sound needs this many onsets
RARE_MAX_K = 4                 # at most this many rare sounds
RARE_SIL = 0.2                 # silhouette for splitting the unexplained onsets into several sounds
DRUM_K_MAX = 8                 # k-means clusters over the onsets (SO.cluster; alike ones merge, small ones drop). 12 and 16
                               # measured 2026-09-09 on the truth set: all-hits F1 0.77 -> 0.80 / 0.81 but the per-sound median
                               # 0.73 -> 0.63 / 0.64 - more clusters give the resolution more wrong merges among the cymbals
DRUM_MIN_SHARE = 0.012         # a cluster under this share of the onsets (and under 6) is dropped
DRUM_ONSETS = "flux"           # "flux": one envelope | "bands": per-band (measured WORSE on the truth set 2026-09-09: onsets F1 0.70 -> 0.59)
DRUM_ONSET_DELTA = 0.03        # flux peak threshold (was 0.12; see _drum_instruments)
DRUM_TAIL_S = 0.11             # the timbre vector's window after the onset (its decay tells a ride from a closed hat)
ONSET_BANDS_HZ = ((0.0, 220.0), (220.0, 2500.0), (2500.0, 20000.0))
ONSET_BAND_DELTA = 0.18        # peak threshold per band, relative to the band's own loud onsets (95th percentile)
ONSET_MERGE_S = 0.012          # onsets from different bands this close are one


def _rare_sounds(Vs, starts, fr, T, alive, W, own, credits, n_mels):
    """Onsets whose patch the alive templates cannot explain (NNLS residual
    >= RARE_UNEXPLAINED of the patch energy) are the sounds the clusterer
    never gave a cluster: their RESIDUAL patches (what is left after the
    alive sounds' fit) are clustered, and each cluster of RARE_MIN or more
    becomes a sound owning those onsets, with the residual's median as its
    template. Returns (alive, W, own, credits) with the new sounds appended
    and their onsets removed from whoever owned or was credited them."""
    from scipy.optimize import nnls
    if not alive:
        return alive, W, own, credits
    A = np.sqrt(W[:, alive])
    cand, resid = [], []
    for i in range(len(starts)):
        f0 = fr(starts[i])
        if f0 + 2 > T:
            continue
        patch = Vs[:, max(0, f0 - 1):f0 + 2].mean(axis=1)
        t = np.sqrt(patch)
        tt = float((t ** 2).sum())
        if tt <= 1e-12:
            continue
        coef, _r = nnls(A, t)
        res = t - A @ coef
        r_pos = float((np.maximum(res, 0.0) ** 2).sum() / tt)
        if r_pos >= RARE_UNEXPLAINED:
            cand.append(i)
            resid.append(np.maximum(res, 0.0))
    if len(cand) < RARE_MIN:
        return alive, W, own, credits
    R = np.array(resid)
    X = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    labels = _cluster_auto(X, RARE_MAX_K, min_share=0.0, min_sil=RARE_SIL, standardize=False) if len(cand) >= 12 else np.zeros(len(cand), dtype=int)
    W = np.array(W, dtype=np.float64)
    next_id = max(own) + 1 if own else 0
    new = []
    for lab in sorted(set(labels.tolist())):
        idx = [cand[q] for q in range(len(cand)) if labels[q] == lab]
        if len(idx) < RARE_MIN:
            continue
        col = np.median(R[labels == lab] ** 2, axis=0) + 1e-9
        W = np.concatenate([W, col[:, None]], axis=1)
        own[next_id] = list(idx)
        credits[next_id] = []
        alive.append(next_id)
        new.append((next_id, set(idx)))
        next_id += 1
    if new:
        taken = set().union(*(s for _k, s in new))
        for k in list(own):
            if k in dict(new):
                continue
            own[k] = [i for i in own[k] if i not in taken]
            credits[k] = [c for c in credits[k] if c[0] not in taken]
    return alive, W, own, credits


def _drum_onsets(y, hop=256):
    """Onsets of the drum stem with one spectral-flux envelope PER BAND
    (low / mid / high), each picked against its own level, merged. One
    full-band envelope (librosa's default) scales with the loudest sounds:
    on the truth set it found 285 of a pop kit's 681 hits - the hats,
    quiet next to the kick and snare, were two thirds missing
    (recall 0.33), and the reading can only cluster what was detected."""
    import librosa
    S = librosa.feature.melspectrogram(y=np.ascontiguousarray(y, dtype=np.float32), sr=RATE, n_fft=1024, hop_length=hop, n_mels=64,
                                       fmin=30.0, fmax=16000.0, power=1.0)
    L = np.log1p(S * 1000.0)
    freqs = librosa.mel_frequencies(n_mels=64, fmin=30.0, fmax=16000.0)
    flux = np.maximum(np.diff(L, axis=1, prepend=L[:, :1]), 0.0)
    wait = max(1, int(0.03 * RATE / hop))
    picked = []
    for lo, hi in ONSET_BANDS_HZ:
        sel = (freqs >= lo) & (freqs < hi)
        if not sel.any():
            continue
        env = flux[sel].mean(axis=0)
        ref = float(np.percentile(env[env > 0], 95)) if np.any(env > 0) else 0.0
        if ref <= 0:
            continue
        env = env / ref
        fr = librosa.onset.onset_detect(onset_envelope=env, sr=RATE, hop_length=hop, units="frames", backtrack=False,
                                        delta=ONSET_BAND_DELTA, wait=wait)
        picked.extend(int(f) * hop for f in fr)
    picked.sort()
    merged = []
    for s in picked:
        if merged and s - merged[-1] < ONSET_MERGE_S * RATE:
            continue
        merged.append(s)
    return merged


DRUM_READER = "drumsep"        # "drumsep": the stem split into kick / snare / toms / cymbals by a learned separator
                               # (lib/dj/drumsep.py) and each family read alone | "mixed": the cluster reader on the whole
                               # stem. Truth set 2026-09-09: all-hits F1 0.77 -> 0.87, per-sound mean 0.62 -> 0.75 (kick
                               # 0.87-1.00 everywhere, toms 0.2 -> 0.5-1.0); the mixed reader is the fallback without the model
DRUMSEP_GATE_DB = 18.0         # inside a family, onsets this far under its loud onsets (95th pct, 60 ms rms) are the other
                               # families bleeding through and are dropped (0 -> 0.74 all-hits, 18 -> 0.87, 24 -> 0.79)
DRUMSEP_FLOOR_DBFS = -60.0     # a family this quiet is not in the kit
_FAMILY_HINTS = {"kick": {"*": "kick"}, "toms": {"*": "tom"}}
DRUM_SEPARATOR = "drumsep"     # "drumsep" (kick / snare / toms / cymbals) | "larsnet" (+ the hi-hat apart from the cymbals;
                               # lib/dj/larsnet.py, checkpoints CC BY-NC)
FAMILY_K_MAX = {"kick": 4, "snare": 3, "toms": 2, "hihat": 2, "cymbals": 3}   # k-means clusters a family may start with
                               # (a family holds a few sounds; eight per family gave LarsNet 32 voices for six sounds).
                               # kick 2 -> 4 (2026-09-09): at 2 the synthetic gate's clap share of the kick family merged
                               # into the kick (precision 0.80); at 4 it reads 1.00 / 1.00, and the truth set is unmoved
                               # (medians 0.87 / 0.69 / 0.72 either way; funk kick 0.76 -> 0.85, house all-hits 0.84 -> 0.77)
FAMILY_SCALAR_WEIGHT = {}           # per family: weight on the timbre vector's two scalar features (decay dB, log-centroid)
                                    # against its ~80 standardised spectral dimensions in the k-means (see _drum_sounds).
                                    # Diagnostic 2026-09-10 on TRUE isolated hits inside the drumsep cymbal stem: ring /
                                    # decay and centroid separate hat from shaker (house: 115 vs 45 ms, 12.0 vs 10.7 kHz)
                                    # and hat from ride (ballad: 177 vs 300 ms, 11.9 vs 10.3 kHz) where the spectra alone
                                    # left them in one voice. {"cymbals": 4.0} measured on both truth sets 2026-09-10 and
                                    # REJECTED: house hat 0.96 -> 0.97, shaker 0.74 -> 0.77 but its clap 0.90 -> 0.73, dnb
                                    # all-hits 0.70 -> 0.67, the ballad's hat and ride still one voice, Slakh unmoved. Empty
FAMILY_TAIL_S = {}                  # the timbre window per family (default DRUM_TAIL_S). {"cymbals": 0.25} measured on the truth
                                    # set 2026-09-09: all-hits 0.87 -> 0.84 (rock 0.90 -> 0.78, dnb 0.82 -> 0.72), per-sound
                                    # median 0.69 -> 0.69; the house hat gained (0.83 -> 0.92), four other hats lost. Off
_CYMBAL_REMAP = {"kick": "crash", "snare": "ride", "tom": "ride"}


def _drum_instruments(y, beats, period, progress=None):
    from lib.gen.analysis import sounds as SO
    if DRUM_READER == "drumsep":
        try:
            return _drum_instruments_split(y, beats, period, progress=progress)
        except Exception as e:  # noqa: BLE001
            if progress:
                progress(f"drums: drumsep reading unavailable ({type(e).__name__}: {str(e)[:60]}); reading the mixed stem")
    if DRUM_ONSETS == "bands":
        starts = _drum_onsets(y)
    else:
        # delta 0.12 -> 0.03 (truth set 2026-09-09): the quiet hats next to the kick were two thirds missing
        # at 0.12 (distinct hit moments recalled 0.67 / 0.56 on two kits); at 0.03 recall 0.85 / 0.87 with
        # precision still 0.99-1.00 - the flux envelope's peaks are clean, the threshold was the loss
        starts, _env = SO.onsets(y, delta=DRUM_ONSET_DELTA, wait_s=0.03)
    if len(starts) < 8:
        return [], len(starts)
    out = _drum_sounds(y, starts, beats, period, progress=progress)
    out.sort(key=lambda d: -d["n"])
    return out, len(starts)


def _family_onsets(y_sub):
    """Onsets of one drumsep family, the other families' bleed gated away."""
    from lib.gen.analysis import sounds as SO
    starts, _env = SO.onsets(y_sub, delta=DRUM_ONSET_DELTA, wait_s=0.03)
    if not starts or DRUMSEP_GATE_DB <= 0:
        return list(starts)
    n = int(0.06 * RATE)
    en = np.array([20 * np.log10(np.sqrt(np.mean(y_sub[s:s + n].astype(np.float64) ** 2)) + 1e-9) if s + 8 < len(y_sub) else -120.0 for s in starts])
    ref = float(np.percentile(en, 95))
    return [int(s) for s, e in zip(starts, en) if e >= ref - DRUMSEP_GATE_DB]


def _drum_instruments_split(y, beats, period, progress=None):
    """Each drumsep family read alone; the exemplar cuts still come from the
    mixed stem (the samples are cut there), isolated against EVERY family's
    onsets. -> (instruments, n_onsets)"""
    if DRUM_SEPARATOR == "larsnet":
        from lib.dj import larsnet as LN
        subs = LN.separate(y, progress=progress)
    else:
        from lib.dj import drumsep as DS
        subs = DS.separate(y, progress=progress)
    fam_starts = {}
    for fam, y_sub in subs.items():
        lvl = 20 * np.log10(np.sqrt(np.mean(y_sub.astype(np.float64) ** 2)) + 1e-9)
        fam_starts[fam] = _family_onsets(y_sub) if lvl >= DRUMSEP_FLOOR_DBFS else []
    all_starts = sorted(s for lst in fam_starts.values() for s in lst)
    if progress:
        progress("drums: drumsep families " + ", ".join(f"{f} {len(s)}" for f, s in fam_starts.items()))
    out = []
    for fam in ("kick", "snare", "toms", "hihat", "cymbals"):
        starts = fam_starts.get(fam) or []
        if len(starts) < 8:
            continue
        insts = _drum_sounds(subs[fam], starts, beats, period, progress=None, y_cut=y, all_starts=all_starts, family=fam,
                             tail_s=FAMILY_TAIL_S.get(fam, DRUM_TAIL_S), k_max=FAMILY_K_MAX.get(fam, DRUM_K_MAX))
        for inst in insts:
            inst["family"] = fam
        if FAMILY_MERGE_TWINS and len(insts) > 1:
            insts, n_tw = _merge_family_twins(insts, beats)
            if progress and n_tw:
                progress(f"drums: {fam}: {n_tw} twin voice(s) merged (the same hits read twice)")
        out.extend(insts)
    if FAMILY_REST:
        # the separator can drop a sound outright (Slakh Track00002: every hat and shaker, +29 dB at their
        # onsets in the drums stem, in NO family stem): the mixed stem's onsets that no family claims within
        # REST_WIN_S are read as a family of their own, from the mixed stem, gated like a family's onsets
        from lib.gen.analysis import sounds as SO
        mixed, _env = SO.onsets(y, delta=DRUM_ONSET_DELTA, wait_s=0.03)
        fam_arr = np.array(all_starts, dtype=np.int64)
        win = int(REST_WIN_S * RATE)
        rest = []
        for s in mixed:
            k = int(np.searchsorted(fam_arr, s))
            if not any(0 <= j < len(fam_arr) and abs(int(fam_arr[j]) - int(s)) <= win for j in (k - 1, k)):
                rest.append(int(s))
        if rest and DRUMSEP_GATE_DB > 0:
            n = int(0.06 * RATE)
            lv = lambda ss: np.array([20 * np.log10(np.sqrt(np.mean(y[s:s + n].astype(np.float64) ** 2)) + 1e-9) if s + 8 < len(y) else -120.0 for s in ss])
            ref = float(np.percentile(lv(mixed), 95))
            rest = [s for s, e in zip(rest, lv(rest)) if e >= ref - DRUMSEP_GATE_DB]
        if progress:
            progress(f"drums: {len(rest)} of the stem's {len(mixed)} onsets claimed by no family")
        if len(rest) >= REST_MIN_ONSETS:
            insts = _drum_sounds(y, rest, beats, period, progress=None, y_cut=y, all_starts=sorted(all_starts + rest), family="rest",
                                 k_max=FAMILY_K_MAX.get("rest", 2))
            for inst in insts:
                inst["family"] = "rest"
            out.extend(insts)
            all_starts = sorted(all_starts + rest)
    n_bleed = 0
    if DROP_FAMILY_BLEED:
        out, n_bleed = _drop_family_bleed(out, beats)
    if out:
        top = max(i["level_dbfs"] for i in out)
        for inst in out:
            inst["level_db"] = round(float(inst["level_dbfs"] - top), 1)
    out.sort(key=lambda d: -d["n"])
    if progress:
        progress(f"drums: {len(out)} sounds over {len(out and set(i['family'] for i in out) or [])} families ({n_bleed} bleed voices dropped)")
    return out, len(all_starts)


FAMILY_REST = False            # the mixed stem's onsets no family claims, read as a "rest" family from the mixed stem.
                               # OFF: measured 2026-09-10 on both truth sets - identical numbers everywhere, because every
                               # mixed-stem onset IS claimed: the separator does not drop Slakh Track00002's hats and
                               # shakers, it routes them into the snare and toms families (cymbals 0, snare 180, toms 106
                               # onsets). A mixed-stem fallback for such kits measured worse (Slakh per sound 0.51 -> 0.39)
REST_WIN_S = 0.02              # a mixed-stem onset within this of any family onset is that family's
REST_MIN_ONSETS = 12           # fewer unclaimed onsets than this are noise, not a sound
DROP_FAMILY_BLEED = False      # OFF: measured 2026-09-09 on the truth set at 0.8/3 dB, 0.9/6 dB and 0.95/3 dB - all-hits
                               # 0.87 -> 0.85-0.86, per-sound mean 0.73 -> 0.72 (it also drops real rare sounds: a crash, a
                               # rim, a clap); it only reduced the voice count on the synthetic gate's additive kit, which
                               # the separator sprays over all four families (446 "toms" onsets on a kit without toms)
BLEED_COINCIDE = 0.8           # a voice whose hits fall within BLEED_WIN_S of another family's voice's hits this often...
BLEED_WIN_S = 0.03
BLEED_QUIETER_DB = 3.0         # ...and that is at least this much quieter than it, is that sound bleeding into this family


def _drop_family_bleed(insts, beats):
    """Cross-family duplicates: the separator leaves some of every sound in
    the other families' stems, and the family reader makes a voice of it.
    Such a voice has the same hit times as a louder voice elsewhere and
    goes. -> (instruments, n_dropped)"""
    if len(insts) < 2:
        return insts, 0
    times = {}
    for k, inst in enumerate(insts):
        times[k] = np.array(sorted(event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0) for e in inst["events"]))
    drop = set()
    # quietest first: a bleed voice is judged against the LOUDER voices of the other families together
    # (a bleed voice often carries two sounds' hits - half kick, half clap - and coincides with neither alone)
    for a in sorted(range(len(insts)), key=lambda k: insts[k]["level_dbfs"]):
        ia = insts[a]
        ta = times[a]
        if len(ta) == 0:
            drop.add(a)
            continue
        louder = [b for b, ib in enumerate(insts) if b != a and b not in drop and ib.get("family") != ia.get("family")
                  and ib["level_dbfs"] - BLEED_QUIETER_DB >= ia["level_dbfs"] and len(times[b])]
        if not louder:
            continue
        tb = np.sort(np.concatenate([times[b] for b in louder]))
        k = np.searchsorted(tb, ta)
        near = np.zeros(len(ta), dtype=bool)
        for j in (k - 1, k):
            jj = np.clip(j, 0, len(tb) - 1)
            near |= np.abs(tb[jj] - ta) <= BLEED_WIN_S
        if near.mean() >= BLEED_COINCIDE:
            drop.add(a)
    return [inst for k, inst in enumerate(insts) if k not in drop], len(drop)


FAMILY_MERGE_TWINS = False     # within a family, two voices whose hits mostly coincide are one sound read twice (LarsNet
                               # 2026-09-09: 12 voices for 6 sounds on the house kit, the duplicates adding false hits).
                               # OFF: measured on the truth set 2026-09-09 - all-hits F1 up (drumsep 0.87 -> 0.90, LarsNet
                               # 0.79 -> 0.85) but PER-SOUND F1 down (drumsep median 0.69 -> 0.66, mean 0.72 -> 0.68;
                               # LarsNet 0.85 -> 0.73): the merges swallow real sounds (ballad snare 1.00 -> 0.67, tom
                               # 0.70 -> 0.48). Same trade as DRUM_K_MAX: fewer voices count more hits, name fewer sounds
TWIN_COINCIDE = 0.6            # ...when this share of the smaller voice's hits fall within TWIN_WIN_S of the larger's
TWIN_WIN_S = 0.02


def _merge_family_twins(insts, beats):
    """Voices of one family whose hits coincide (TWIN_COINCIDE of the smaller's within TWIN_WIN_S of the
    larger's) merge into the larger: the union of hits, each time once. -> (instruments, n_merged)"""
    def times_of(inst):
        return np.array(sorted(event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0) for e in inst["events"]))
    insts = sorted(insts, key=lambda i: -i["n"])
    merged = 0
    changed = True
    while changed and len(insts) > 1:
        changed = False
        for a in range(len(insts)):
            ta = times_of(insts[a])
            for b in range(len(insts) - 1, a, -1):
                tb = times_of(insts[b])
                if len(tb) == 0 or len(ta) == 0:
                    continue
                k = np.searchsorted(ta, tb)
                near = np.zeros(len(tb), dtype=bool)
                for j in (k - 1, k):
                    jj = np.clip(j, 0, len(ta) - 1)
                    near |= np.abs(ta[jj] - tb) <= TWIN_WIN_S
                if near.mean() >= TWIN_COINCIDE:
                    # the larger keeps its recording; the smaller's hits that are new join it
                    extra = [e for e, n_ in zip(insts[b]["events"], near) if not n_]
                    insts[a]["events"] = sorted(insts[a]["events"] + extra, key=lambda e: (e[0], e[1]))
                    insts[a]["n"] = len(insts[a]["events"])
                    del insts[b]
                    merged += 1
                    changed = True
                    break
            if changed:
                break
    return insts, merged


def _drum_sounds(y, starts, beats, period, progress=None, y_cut=None, all_starts=None, family=None, tail_s=None, k_max=None):
    """The cluster reader over given onsets of signal y (a stem or one drumsep family).
    y_cut: the signal the exemplar cuts refer to (default y); all_starts: every onset the cuts
    must stay clear of (default starts). -> instruments (unsorted, level_db relative within)."""
    from lib.gen.analysis import sounds as SO
    y_cut = y if y_cut is None else y_cut
    X, meta = SO.timbre(y, starts, attack_s=0.02, body_s=0.06, tail_s=tail_s or DRUM_TAIL_S)
    Xn = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    w_sc = FAMILY_SCALAR_WEIGHT.get(family or "", 1.0)
    if w_sc != 1.0:
        # the two scalar features (decay, log-centroid) are 2 of ~80 standardised dimensions and count for
        # nothing next to the spectra; in the cymbal family they are what tells a hat from a shaker from a ride
        Xn = Xn.copy()
        Xn[:, -2:] *= w_sc
    labels, keep = SO.cluster(Xn, k_max=k_max or DRUM_K_MAX, min_share=DRUM_MIN_SHARE)
    if progress:
        progress(f"drums: {len(starts)} onsets, {len(keep)} sound clusters")
    V = SO._mel(y)
    hop = SO.HOP
    # templates are PATCHES, not spectra: the mel frame at the onset and the frames 23 and 58 ms
    # later stacked, so a sound is its spectrum AND its envelope (a shaker's soft attack and a
    # hat's instant one are different sounds with the same spectrum; an open hat rings where a
    # closed one has stopped)
    T = V.shape[1] - max(_TEMPLATE_OFFSETS)
    Vs = np.concatenate([V[:, o:o + T] for o in _TEMPLATE_OFFSETS], axis=0)
    fr = lambda s: int(s / hop)

    def template(idx):
        cols = [Vs[:, max(0, fr(starts[i]) - 1):fr(starts[i]) + 2].mean(axis=1) for i in idx[:300] if fr(starts[i]) + 2 <= T]
        return np.median(cols, axis=0) if cols else np.full(Vs.shape[0], 1e-6)
    own = {k: [int(i) for i in np.where(labels == lab)[0]] for k, lab in enumerate(keep)}
    W = np.array([template(own[k]) for k in own]).T
    n_mels = len(SO._mel_freqs())
    own_all = {k: list(v) for k, v in own.items()}          # every cluster's onsets, before resolution
    alive, W, own, credits, own_frac = _resolve_clusters(W, own, template, n_mels)
    if ONSET_CREDITS:
        # every onset explained by the alive sounds' templates: a hat under a kick lands in the KICK
        # cluster (the kick owns the timbre) and never forms a coincidence cluster to resolve, so the
        # cluster-level credits above never see it. Truth set 2026-09-09: the hats ON the kick beats
        # were the missing ones (recall 0.00-0.23 loud vs 0.77 soft/alone)
        credits = _onset_credits(Vs, starts, fr, T, alive, W, own, credits, n_mels)
    if RARE_SOUNDS:
        n_before = len(alive)
        alive, W, own, credits = _rare_sounds(Vs, starts, fr, T, alive, W, own, credits, n_mels)
        if progress and len(alive) > n_before:
            progress(f"drums: {len(alive) - n_before} rare sound(s) from unexplained onsets")
    sounds, hits = [], {}
    for q, k in enumerate(alive):
        idx = np.array(sorted(set(own[k])), dtype=int)
        own_db = 0.0
        lv = np.array([meta[i][2] for i in idx])
        s, e = SO._exemplar(y_cut, [starts[i] for i in idx], [meta[i][2] for i in idx], all_starts=all_starts if all_starts is not None else starts)
        dur = (e - s) / RATE
        # the LEVEL REFERENCE is the exemplar's own hit: the sample is that recording, unscaled, and every
        # event's velocity is its onset energy relative to it (measured the same way on the same
        # window) - no cluster percentile, no energy-fraction guess for the sound's own hits
        ex_i = next((i for i in idx if int(starts[i]) == s), None)
        top = float(meta[ex_i][2]) if ex_i is not None else float(np.percentile(lv, 90))
        lst = {int(starts[i]): (float(np.clip(1.0 + (meta[i][2] - top) / 24.0, 0.2, 1.7)), 1.0) for i in idx}
        # credited coincidences: a hit of this sound unless the sound struck within its own length
        # just before (then what the coincidence cluster heard was its tail, not a new hit); the
        # onset's energy counts by this sound's share of the explanation
        own_t = np.array(sorted(lst), dtype=np.int64)
        n_cred = 0
        share_of, src_of = {}, {}
        for o, sh, src_c in credits[k]:
            if sh >= share_of.get(o, 0.0):
                share_of[o], src_of[o] = sh, src_c
        # a credited hit carries its energy FRACTION of the coincidence onset (relative to this sound's
        # exemplar hit); a cluster-relative variant was measured worse on the synthetic gate (2026-09-09)
        for i in sorted(set(share_of) - set(own[k]), key=lambda i: starts[i]):
            s0 = int(starts[i])
            kk = int(np.searchsorted(own_t, s0))
            prev = int(own_t[kk - 1]) if kk > 0 else -10 ** 9
            if (s0 - prev) / RATE < 0.8 * dur:
                continue
            en = meta[i][2] + 10.0 * np.log10(max(share_of[i], 0.02))
            lst[s0] = (float(np.clip(1.0 + (en - top) / 24.0, 0.2, 1.7)), float(min(0.99, max(share_of[i], 0.02))))
            n_cred += 1
        sounds.append({"idx": idx, "n": int(len(lst)), "centroid": float(np.median([meta[i][0] for i in idx])),
                       "decay": float(np.median([meta[i][1] for i in idx])), "level": float(np.percentile(lv, 75)), "own_db": own_db, "top": top,
                       "cut": (s, e), "dur": dur, "credited": n_cred,
                       "low": float(W[:n_mels, k][SO._mel_freqs() < 200.0].sum() / max(float(W[:n_mels, k].sum()), 1e-9))})
        hits[q] = sorted((s, v) for s, v in lst.items())
    if progress:
        progress(f"drums: {len(sounds)} sounds ({len(keep) - len(alive)} clusters merged or explained as coincidences)")
    hints = SO._assign_slots(sounds, _bar_grids(beats, hits, period))
    if family in _FAMILY_HINTS:
        hints = {i: _FAMILY_HINTS[family]["*"] for i in range(len(sounds))}
    elif family == "snare":
        # the family's main sound is the snare; the slot assigner names the lowest sound "kick"
        main = max(range(len(sounds)), key=lambda i: sounds[i]["n"]) if sounds else None
        hints = {i: ("snare" if i == main else ("perc" if hints.get(i) in (None, "kick", "snare", "hat", "tom") else hints.get(i))) for i in range(len(sounds))}
    elif family == "cymbals":
        hints = {i: _CYMBAL_REMAP.get(hints.get(i), hints.get(i)) for i in range(len(sounds))}
    elif family == "hihat":
        # the busiest sound is the closed hat; the others open hats (or whatever rings longer)
        main = max(range(len(sounds)), key=lambda i: sounds[i]["n"]) if sounds else None
        hints = {i: ("hat" if i == main else "ohat") for i in range(len(sounds))}
    levels = _level_rel([sd["level"] for sd in sounds])
    out = []
    for i, sd in enumerate(sounds):
        # conf marks provenance: 1.0 = the sound's own onset, < 1 = credited from a coincidence (its
        # energy fraction) - the kit extractor cuts samples from the own ones only
        events = _fold_hits(beats, [(s, None, sd["dur"], v[0], v[1]) for s, v in hits[i]], period)
        if not events:
            continue
        s, e = sd["cut"]
        k_alive = alive[i]
        out.append({"stem": "drums", "kind": "hit", "pitched": False, "range": None,
                    "centroid_hz": round(sd["centroid"], 1), "decay_db": round(sd["decay"], 1), "dur_s": round(float(sd["dur"]), 3),
                    "n": int(len(events)), "level_db": levels[i], "exemplar": [round(s / RATE, 4), round(e / RATE, 4)],
                    # the sound's own mel template (48 bands x 3 time blocks, coincidences explained away): masks are built from it
                    "template": [round(float(x), 6) for x in W[:, k_alive]],
                    "level_dbfs": round(float(sd["top"]), 1),         # the exemplar hit's own level
                    "hint": hints.get(i), "events": events})
    return out


def _bar_grids(beats, hits, period):
    """{sound: [(sample, strength)]} -> per bar {sound: [16 strengths]} (bars = 4 beats from beat 0)."""
    n_bars = max(0, len(beats) // 4)
    out = [{i: [0.0] * 16 for i in hits} for _ in range(n_bars)]
    for i, lst in hits.items():
        for s, v in lst:
            v = v[0] if isinstance(v, tuple) else v
            pos = beat_step_of(beats, s / RATE, period)
            if pos is None:
                continue
            bar, st = pos[0] // 4, (pos[0] % 4) * STEPS + pos[1]
            if bar < n_bars:
                out[bar][i][st] = max(out[bar][i][st], float(v))
    return out


# --------------------------------------------------------------------------
# PLUCKED / STRUCK instruments of a pitched stem
# --------------------------------------------------------------------------

def _pluck_instruments(y, P, beats, period, stem, progress=None):
    """Onsets of the percussive part; pitched ones cluster on their
    harmonic profile, unpitched ones on mel timbre."""
    from lib.gen.analysis import sounds as SO
    starts, _env = SO.onsets(P, delta=0.2, wait_s=0.05)
    if len(starts) < 8:
        return [], len(starts)
    X, meta = SO.timbre(P, starts, attack_s=0.02, body_s=0.08, tail_s=0.16)
    lo, hi = PITCH_RANGE[stem]
    pitches = [_onset_pitch(y, s, lo, hi, post_s=POST_S[stem]) for s in starts]
    pitched_idx = [i for i, pc in enumerate(pitches) if pc[0] is not None and pc[1] >= HIT_PITCH_CONF]
    unpitched_idx = [i for i in range(len(starts)) if i not in set(pitched_idx)]
    if progress:
        progress(f"{stem}: {len(starts)} onsets, {len(pitched_idx)} with a pitch")
    groups = []                               # (indices, pitched?)
    if len(pitched_idx) >= max(4, PITCHED_SHARE * len(starts)) or len(pitched_idx) >= 12:
        # dB features on their own scale: two plucks are different instruments when their
        # harmonic profiles differ by PROFILE_SEP dB rms, not when a 1 dB drift is consistent
        F = np.array([np.concatenate([pitches[i][2], [meta[i][1]]]) for i in pitched_idx])
        lab = _cluster_auto(F, MAX_HIT_INSTRUMENTS[stem], min_share=0.04, min_sil=SILHOUETTE_HITS, standardize=False, min_sep=PROFILE_SEP)
        for L in sorted(set(lab.tolist())):
            groups.append(([pitched_idx[j] for j in np.where(lab == L)[0]], True))
    else:
        unpitched_idx = list(range(len(starts)))
    if len(unpitched_idx) >= 12:
        F = X[unpitched_idx]
        lab = _cluster_auto(F, 3, min_share=0.08)
        for L in sorted(set(lab.tolist())):
            groups.append(([unpitched_idx[j] for j in np.where(lab == L)[0]], False))
    env, hop = _rms_env(y)
    sorted_starts = np.asarray(sorted(starts), dtype=np.int64)
    out = []
    levels = []
    for idx, pitched in groups:
        if len(idx) < 4:
            continue
        lv = np.array([meta[j][2] for j in idx])
        top = float(np.percentile(lv, 90))
        items, durs = [], []
        n_ghost = 0
        smooth = {}
        if pitched and CONTINUITY:
            # measured on three real tracks (2026-09-08): the continuity prior LOWERED chroma agreement
            # (bass lines leap; the prior pulled true leaps onto ghosts) - off unless asked for
            order = sorted((j for j in idx if pitches[j][0] is not None), key=lambda j: starts[j])
            path = _viterbi_pitches([(starts[j] / RATE, pitches[j][3]) for j in order])
            smooth = {j: m for j, m in zip(order, path)}
        for j in idx:
            if meta[j][2] < top - HIT_FLOOR_DB:
                n_ghost += 1                 # far under the instrument's own hits: bleed, not a note
                continue
            s0 = int(starts[j])
            k = int(np.searchsorted(sorted_starts, s0, side="right"))
            nxt = int(sorted_starts[k]) if k < len(sorted_starts) else len(y)
            d = _hit_duration_s(env, hop, s0, nxt)
            durs.append(d)
            m, c, _p = pitches[j][:3]
            if pitched and j in smooth:
                m = smooth[j]
            if not pitched:
                m, c = None, 0.0
            items.append((s0, m, d, float(np.clip(1.0 + (meta[j][2] - top) / 24.0, 0.2, 1.5)), c))
        events = _fold_hits(beats, items, period)
        if not events:
            continue
        s, e = SO._exemplar(y, [starts[j] for j in idx], [meta[j][2] for j in idx], all_starts=starts)
        ex_midi = next((smooth.get(j, pitches[j][0]) for j in idx if int(starts[j]) == s), None) if pitched else None
        ms = [m for _s, m, _d, _v, _c in items if m is not None]
        cen = float(np.median([meta[j][0] for j in idx]))
        dur = float(np.median(durs))
        out.append({"stem": stem, "kind": "hit", "pitched": bool(pitched), "range": [int(min(ms)), int(max(ms))] if ms else None,
                    "centroid_hz": round(cen, 1), "decay_db": round(float(np.median([meta[j][1] for j in idx])), 1), "dur_s": round(dur, 3),
                    "n": int(len(events)), "ghosts": int(n_ghost), "exemplar": [round(s / RATE, 4), round(e / RATE, 4)],
                    "level_dbfs": round(top, 1),
                    "exemplar_midi": (int(ex_midi) if ex_midi is not None else None),
                    "hint": _pitched_hint(stem, pitched, cen, dur), "events": events})
        levels.append(float(np.percentile(lv, 75)))
    for inst, lv in zip(out, _level_rel(levels)):
        inst["level_db"] = lv
    out.sort(key=lambda d: -d["n"])
    return out, len(starts)


def _pitched_hint(stem, pitched, centroid, dur):
    if stem == "vocals":
        return "voice" if pitched else "vocal hit"
    if stem == "bass":
        return "bass" if pitched else "low perc"
    if not pitched:
        return "perc"
    return "stab" if dur < 0.25 else "pluck"


# --------------------------------------------------------------------------
# SUSTAINED instruments (pitched stems)
# --------------------------------------------------------------------------

def _struck_table(hit_insts, n_beats, grace_beats=1):
    """(beat, midi) pairs where a pitched HIT instrument struck that note
    and it is still sounding (its measured length, plus a beat of grace
    for the room): those holds belong to the hit, not to a pad."""
    struck = set()
    for inst in hit_insts or []:
        if not inst.get("pitched"):
            continue
        for b, st, m, dur, _v, _c in (e[:6] for e in inst["events"]):
            if m is None:
                continue
            last = b + int(np.ceil((st + dur) / STEPS)) - 1 + grace_beats
            for k in range(b, min(n_beats, last + 1)):
                struck.add((k, int(m)))
    return struck


def _sustain_instruments(y, H, beats, period, stem, hit_insts=None, progress=None):
    """Per-beat held notes on the harmonic part - only the notes NOBODY
    STRUCK (a lead's long note is the lead's, with its measured length;
    what remains is the pad, the drone, the sung line) - clustered into
    instruments by register, polyphony and spectral envelope."""
    import librosa
    lo, hi = PITCH_RANGE[stem]
    poly = POLY[stem]
    n = len(beats)
    if n < 8:
        return []
    ends = np.concatenate([beats[1:], [beats[-1] + period]])
    rms = np.zeros(n)
    hold = np.zeros(n)
    for k in range(n):
        a, b = int(beats[k] * RATE), int(min(len(H), ends[k] * RATE))
        if b - a < 1024:
            continue
        seg = H[a:b]
        rms[k] = float(np.sqrt(np.mean(seg ** 2)))
        q = (b - a) * 2 // 5
        head = float(np.sqrt(np.mean(seg[:q] ** 2))) + 1e-9
        tail = float(np.sqrt(np.mean(seg[-q:] ** 2))) + 1e-9
        hold[k] = tail / head
    loud = rms[rms > 0]
    if len(loud) < 8:
        return []
    ref = float(np.percentile(loud, 95))
    floor = ref * 10 ** (SUSTAIN_FLOOR_DB / 20.0)
    active = [k for k in range(n) if rms[k] >= floor and hold[k] >= SUSTAIN_HOLD]
    if len(active) < 8:
        return []
    if progress:
        progress(f"{stem}: {len(active)} held beats")
    notes, feats = {}, []
    half2 = {}
    struck = _struck_table(hit_insts, n)
    n_struck = 0
    for k in active:
        a, b = int(beats[k] * RATE), int(min(len(H), ends[k] * RATE))
        seg = H[a:b]
        mid = len(seg) // 2
        if mid < 512:
            continue
        Ra, df = _spectrum(seg[:mid])
        Rb, df_b = _spectrum(seg[mid:])
        if abs(df - df_b) > 1e-9:                       # equal halves have equal FFT sizes; guard the odd sample
            Rb, df_b = _spectrum(seg[mid:2 * mid])
        sa, _ = _harmonic_salience(Ra, df, lo, hi)
        sb, _ = _harmonic_salience(Rb, df, lo, hi)
        half2[k] = sb                                      # the beat's second half: where a hold may have BEGUN
        sal = np.minimum(sa, sb)
        if float(sal.max()) < SUSTAIN_TONAL * (float(np.median(sal)) + 1e-12):
            continue                                        # nothing tonal holds through this beat
        # prominent in BOTH halves: a tail that fades or a note that only enters mid-beat is not held
        sal[sal < SUSTAIN_PROMINENT * max(float(sa.max()), float(sb.max()), 1e-12)] = 0.0
        picked = [(m, c) for m, c in _pick_notes(sal, lo, poly) if (k, m) not in struck]
        if not picked:
            n_struck += 1
        if not picked:
            continue
        notes[k] = picked
        mf = librosa.feature.mfcc(y=np.ascontiguousarray(seg), sr=RATE, n_mfcc=14, n_fft=2048, hop_length=1024).mean(axis=1)[1:]
        S = librosa.feature.melspectrogram(y=np.ascontiguousarray(seg), sr=RATE, n_fft=2048, hop_length=1024, n_mels=40,
                                           fmin=30.0, fmax=12000.0, power=1.0).mean(axis=1)
        L = 20.0 * np.log10(S + 1e-6)
        L -= L.max()
        w = 10 ** (L / 20.0)
        cen = float((w * librosa.mel_frequencies(n_mels=40, fmin=30.0, fmax=12000.0)).sum() / max(float(w.sum()), 1e-9))
        # register + polyphony weigh as much as the envelope: a held lead line and a pad
        # chord on the same synth are different instruments to the ear
        mean_midi = float(np.mean([m for m, _c in picked]))
        feats.append((k, mf, cen, mean_midi, len(picked)))
    if len(feats) < 8:
        return []
    ks = [f[0] for f in feats]
    M = np.array([f[1] for f in feats])
    cens = np.array([f[2] for f in feats])
    # envelope as z-scores, then register (an octave = 3 units) and polyphony (a note = 2 units)
    # on their own scales, so a held lead line and a pad chord on one synth part ways
    Mz = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-6)
    F = np.concatenate([Mz, np.array([[f[3] / 4.0, f[4] * 2.0] for f in feats])], axis=1)
    labels = _cluster_auto(F, MAX_SUSTAIN_INSTRUMENTS[stem], min_share=0.06, standardize=False)
    if progress and n_struck:
        progress(f"{stem}: {n_struck} held beats were struck notes ringing")
    out = []
    for lab in sorted(set(labels.tolist())):
        sel = [ks[j] for j in np.where(labels == lab)[0]]
        selset = set(sel)
        events = []
        open_notes = {}                   # midi -> [beat0, n_beats, vel_sum, conf_sum, early, [vel per beat]]
        prev = None
        poly_beats = 0
        hold_slopes = []                  # dB/s of the level over each hold of 3+ beats (the voice's own decay)

        def flush(m):
            b0, nb, vs, cs, early, vels = open_notes.pop(m)
            # a hold that was already prominent in the previous beat's second half began there:
            # start it two steps earlier (the reader only counts beats it holds THROUGH)
            if early:
                events.append([b0 - 1, 2, m, nb * STEPS + 2, round(vs / nb, 3), round(cs / nb, 3), 0.0])
            else:
                events.append([b0, 0, m, nb * STEPS, round(vs / nb, 3), round(cs / nb, 3), 0.0])
            if nb >= 3:
                xs = np.arange(nb) * period
                hold_slopes.append(float(np.polyfit(xs, (np.array(vels) - 1.0) * 30.0, 1)[0]))
        for k in sorted(sel):
            if prev is not None and k != prev + 1:
                for m in list(open_notes):
                    flush(m)
            cur = {m: c for m, c in notes[k]}
            if len(cur) > 1:
                poly_beats += 1
            vel = float(np.clip(20 * np.log10(rms[k] / max(ref, 1e-9)) / 30.0 + 1.0, 0.15, 1.0))
            for m in list(open_notes):
                if m not in cur:
                    flush(m)
            for m, c in cur.items():
                if m in open_notes:
                    open_notes[m][1] += 1
                    open_notes[m][2] += vel
                    open_notes[m][3] += c
                    open_notes[m][5].append(vel)
                else:
                    prev_half = half2.get(k - 1)
                    early = bool(prev_half is not None and (k - 1) not in selset and prev_half.max() > 0
                                 and prev_half[m - lo] >= SUSTAIN_PROMINENT * float(prev_half.max()))
                    open_notes[m] = [k, 1, vel, c, early, [vel]]
            prev = k
        for m in list(open_notes):
            flush(m)
        events.sort()
        if not events:
            continue
        ms = [e[2] for e in events]
        best_k, best_v = sel[0], -1.0
        for k in sel:
            if k + 1 in selset and rms[k] + rms[k + 1] > best_v:
                best_k, best_v = k, rms[k] + rms[k + 1]
        ex_end = float(ends[best_k + 1] if best_k + 1 < n else ends[best_k])
        cen = float(np.median(cens[labels == lab]))
        out.append({"stem": stem, "kind": "sustain", "pitched": True, "range": [int(min(ms)), int(max(ms))],
                    "centroid_hz": round(cen, 1), "decay_db": 0.0, "dur_s": round(float(np.mean([e[3] for e in events]) / STEPS * period), 3),
                    "n": int(len(sel)), "level": float(np.percentile([rms[k] for k in sel], 75)),
                    "exemplar": [round(float(beats[best_k]), 4), round(ex_end, 4)],
                    "exemplar_midi": int(min(m for m, _c in notes[best_k])),
                    "level_dbfs": round(float(20 * np.log10(ref + 1e-9)), 1),
                    "hold_decay_db_s": (round(float(np.median(hold_slopes)), 2) if hold_slopes else None),
                    "hint": _sustain_hint(stem, cen, poly_beats / max(1, len(sel))), "events": events})
    levels = _level_rel([20 * np.log10(d["level"] + 1e-9) for d in out])
    for d, lv in zip(out, levels):
        d["level_db"] = lv
        del d["level"]
    out.sort(key=lambda d: -d["n"])
    return out


def _sustain_hint(stem, centroid, poly_share):
    if stem == "bass":
        return "sub" if centroid < 120 else "bass hold"
    if stem == "vocals":
        return "sung line"
    return "pad" if poly_share >= 0.5 else "lead"


# --------------------------------------------------------------------------
# Closed-loop pitch verification (monophonic stems)
# --------------------------------------------------------------------------
BASS_READER = "pitchtrack"     # "pitchtrack" (lib/dj/bassreader.py) | "onsets" (the cluster reader below)
OTHER_READER = "ymt3"          # "ymt3" (YourMT3+, notes labelled by instrument, lib/dj/ymt3.py; falls back to "transcription"
                               # when its interpreter is missing) | "transcription" (basic-pitch + timbre voices,
                               # lib/dj/polyreader.py) | "onsets" (the cluster reader). Truth set 2026-09-09: ymt3 + cross-voice
                               # de-duplication notes F1 0.73 / purity 0.73 against 0.60 / 0.47 for the transcription reader
SUSTAIN_FUSION = False         # with the transcription reader: held notes the transcriber missed, from the sustain pass -
                               # measured neutral on the truth set (pad recall 0.11 -> 0.11: the pitch classes are already
                               # covered by the transcriber's fragments, the missing part is the ONSET of the hold)


def _fuse_holds(y, H, beats, period, stem, insts, progress=None):
    """Held notes (the per-beat sustain reader) on beats where no transcribed
    note of that pitch class is sounding -> one or more sustain voices."""
    sus = _sustain_instruments(y, H, beats, period, stem, hit_insts=insts, progress=None)
    if not sus:
        return []
    # (beat, pitch class) covered by a transcribed note
    covered = set()
    for inst in insts:
        for e in inst["events"]:
            if e[2] is None:
                continue
            n_beats = max(1, int(np.ceil((e[1] + e[3]) / STEPS)))
            for k in range(e[0], e[0] + n_beats):
                covered.add((k, int(e[2]) % 12))
    out = []
    for inst in sus:
        keep = []
        for e in inst["events"]:
            n_beats = max(1, int(np.ceil((e[1] + e[3]) / STEPS)))
            span = [(k, int(e[2]) % 12) for k in range(e[0], e[0] + n_beats)]
            if sum(1 for s in span if s in covered) <= 0.5 * len(span):
                keep.append(e)
        if len(keep) >= 8:
            inst = dict(inst, events=keep, n=len(keep), reader="sustain-fusion")
            ms = [e[2] for e in keep]
            inst["range"] = [int(min(ms)), int(max(ms))]
            out.append(inst)
    if progress and out:
        progress(f"{stem}: {sum(i['n'] for i in out)} held notes the transcription missed, in {len(out)} voice(s)")
    return out
VERIFY_STEMS = ("bass",)
VERIFY_MARGIN = 1.3            # another pitch must be this much more salient over the WHOLE note to replace the read one


def verify_pitches(y, events, beats, period, lo, hi, margin=VERIFY_MARGIN):
    """Re-check every pitched event against the stem over the note's own
    length (30 ms after the onset to its end, at most 0.6 s): harmonic
    salience over the semitone grid; the read pitch is replaced when
    another is `margin` times more salient. The onset reader's short
    window and drone subtraction misread sustained bass notes; the whole
    note does not. -> (events, n_changed)"""
    y = np.asarray(y, dtype=np.float32)
    out = []
    changed = 0
    for ev in events:
        if ev[2] is None:
            out.append(ev)
            continue
        t = event_time(beats, ev[0], ev[1]) + (ev[6] if len(ev) > 6 else 0.0)
        a = int((t + 0.03) * RATE)
        b = int(min(len(y), (t + min(0.6, max(0.12, ev[3] * period / STEPS))) * RATE))
        if b - a < 2048:
            out.append(ev)
            continue
        R, df = _spectrum(y[a:b])
        sal, _ = _harmonic_salience(R, df, lo, hi)
        if sal.max() <= 0:
            out.append(ev)
            continue
        m_read = int(ev[2])
        s_read = float(sal[m_read - lo]) if lo <= m_read <= hi else 0.0
        best = lo + int(np.argmax(sal))
        s_best = float(sal.max())
        if best != m_read and s_best >= margin * max(s_read, 1e-12):
            # the octave check the onset reader uses: the fundamental has odd harmonics
            f0 = 440.0 * 2 ** ((best - 69) / 12.0)
            pk = lambda f: float(R[int(round(f / df))]) if int(round(f / df)) < len(R) else 0.0
            if pk(f0) + pk(3 * f0) + pk(5 * f0) < 0.3 * (pk(2 * f0) + pk(4 * f0) + pk(6 * f0)) and best + 12 <= hi:
                best += 12
            e = list(ev)
            e[2] = int(best)
            out.append(e)
            changed += 1
        else:
            out.append(ev)
    return out, changed


# --------------------------------------------------------------------------
# Polyphony: the transcriber's pitches on the reader's onsets
# --------------------------------------------------------------------------
POLY_STEMS = ("other",)
CONTINUITY = False             # Viterbi pitch continuity for plucked lines (see _pluck_instruments)
POLY_AMP = 0.3                 # a transcribed note this strong counts (its own 0..1 amplitude)
POLY_HOLD_S = 0.35             # a transcribed note this long, not on a hit, is a held note
POLY_WIN = (0.04, 0.06)        # a note starting within (-40 ms, +60 ms) of a hit belongs to it


TRANSCRIBE_HOP_S = 256.0 / 22050.0     # basic-pitch's frame hop (11.6 ms)
TRANSCRIBE_ONSET = 0.5                 # basic-pitch's onset threshold (its default 0.5)
TRANSCRIBE_FRAME = 0.3                 # ...frame threshold (default 0.3). Lower measured on the truth set 2026-09-09: 0.3/0.2 ->
                                       # `other` notes F1 0.62 -> 0.48 (precision 0.59 -> 0.44), 0.4/0.25 -> 0.54; pad recall
                                       # 0.08 -> 0.15 / 0.11, strings 0.28 -> 0.25 / 0.32: the held parts' missing onsets are
                                       # not under the threshold, and everything else that comes up is false
TRANSCRIBE_MIN_MS = 127.7              # ...minimum note length (default 127.7 ms)


def transcribe_full(y):
    """basic-pitch on a mono signal -> (notes [(start_s, end_s, midi, amp)], onset posteriorgram
    [frames, 88] float32 at TRANSCRIBE_HOP_S with midi = index + 21) or (None, None) when unavailable.
    The posteriorgram is the transcriber's own onset detector: where it is low at a note's start,
    the note was made from frame activity (a hold re-triggered), not from an attack."""
    import logging
    root_logger = logging.getLogger()
    prev_level = root_logger.level
    try:
        # basic-pitch logs "Coremltools / tflite-runtime / Tensorflow is not installed" on import; it runs on
        # the ONNX runtime here, so those are noise in every console that reads a track
        root_logger.setLevel(logging.ERROR)
        from basic_pitch.inference import predict
    except Exception:  # noqa: BLE001
        return None, None
    finally:
        root_logger.setLevel(prev_level)
    import os
    import tempfile
    import soundfile as sf
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(path, np.asarray(y, dtype=np.float32), RATE)
        model_out, _midi, notes = predict(path, onset_threshold=TRANSCRIBE_ONSET, frame_threshold=TRANSCRIBE_FRAME,
                                          minimum_note_length=TRANSCRIBE_MIN_MS)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    onset = np.asarray(model_out.get("onset"), dtype=np.float32) if isinstance(model_out, dict) and "onset" in model_out else None
    return sorted((float(s), float(e), int(m), float(a)) for s, e, m, a, _b in notes), onset


def transcribe(y):
    """basic-pitch note events for a mono signal: [(start_s, end_s, midi, amp)] or None when unavailable."""
    return transcribe_full(y)[0]


def _poly_refine(y, insts, beats, period, stem, progress=None):
    """Replace the stem's pitches with a multi-pitch transcription fused
    on the reader's onsets: every transcribed note starting on a hit
    becomes an event of that hit (a chord is several events on one
    step); long transcribed notes off the hits become the held voice,
    with exact starts and lengths. -> (instruments, reason | None)."""
    notes = transcribe(y)
    if notes is None:
        return insts, f"{stem}: basic-pitch not installed - per-onset pitches kept"
    notes = [n for n in notes if n[3] >= POLY_AMP]
    # the transcriber re-triggers a held note: same pitch, a short gap -> one note
    merged = []
    for n in sorted(notes, key=lambda n: (n[2], n[0])):
        if merged and merged[-1][2] == n[2] and n[0] - merged[-1][1] < 0.08:
            s, e, m, a = merged[-1]
            merged[-1] = (s, max(e, n[1]), m, max(a, n[3]))
        else:
            merged.append(n)
    notes = sorted(merged)
    if len(notes) < 8:
        return insts, None
    starts = np.array([n[0] for n in notes])
    hit_dur = {k: float(i.get("dur_s") or 0.25) for k, i in enumerate(insts)}
    lo, hi = PITCH_RANGE[stem]
    used = np.zeros(len(notes), dtype=bool)
    out = []
    n_chord = 0
    for k_inst, inst in enumerate(insts):
        if inst["kind"] != "hit" or not inst.get("pitched"):
            out.append(inst)
            continue
        new_events = []
        for ev in inst["events"]:
            t = event_time(beats, ev[0], ev[1]) + (ev[6] if len(ev) > 6 else 0.0)
            i0 = int(np.searchsorted(starts, t - POLY_WIN[0]))
            i1 = int(np.searchsorted(starts, t + POLY_WIN[1]))
            # a hit takes the notes that start with it AND are hit-length: a pad note beginning
            # under a pluck is the pad's, not a chord tone of the pluck
            max_len = max(0.6, 2.5 * hit_dur.get(k_inst, 0.25))
            here = [i for i in range(i0, i1) if not used[i] and lo <= notes[i][2] <= hi and (notes[i][1] - notes[i][0]) <= max_len]
            if not here:
                new_events.append(ev)
                continue
            here.sort(key=lambda i: -notes[i][3])
            top = notes[here[0]][3]
            for i in here:
                if notes[i][3] < 0.5 * top:
                    continue
                used[i] = True
                dur = max(1, int(round((notes[i][1] - notes[i][0]) / (period / STEPS))))
                e = list(ev)
                e[2] = int(notes[i][2])
                e[3] = int(min(dur, 64))
                new_events.append(e)
            n_chord += max(0, len(here) - 1)
        seen = set()
        dedup = []
        for e in new_events:
            k = (e[0], e[1], e[2])
            if k not in seen:
                seen.add(k)
                dedup.append(e)
        inst = dict(inst, events=sorted(dedup, key=lambda e: (e[0], e[1], -1 if e[2] is None else e[2])))
        ms = [e[2] for e in inst["events"] if e[2] is not None]
        inst["range"] = [int(min(ms)), int(max(ms))] if ms else inst.get("range")
        inst["n"] = len(inst["events"])
        out.append(inst)
    # held notes: the reader's held voices keep their ACTIVITY and levels (where and how loud
    # something holds - measured on the harmonic energy), the transcriber supplies the PITCHES:
    # every held event is re-voiced with the transcribed notes sounding through its span
    n_hold_fixed = 0
    fixed_sus = []
    for inst in [i for i in out if i["kind"] == "sustain"]:
        new_events = []
        for ev in inst["events"]:
            t_a = event_time(beats, ev[0], ev[1]) + (ev[6] if len(ev) > 6 else 0.0)
            t_b = t_a + ev[3] * period / STEPS
            over = [n for n in notes if n[0] < t_b - 0.05 and n[1] > t_a + 0.05 and lo <= n[2] <= hi
                    and (n[1] - n[0]) >= POLY_HOLD_S and min(n[1], t_b) - max(n[0], t_a) >= 0.4 * (t_b - t_a)]
            if not over:
                new_events.append(ev)
                continue
            over.sort(key=lambda n: -n[3])
            top = over[0][3]
            for n in over[:POLY[stem] + 1]:
                if n[3] < 0.45 * top:
                    break
                e = list(ev)
                e[2] = int(n[2])
                new_events.append(e)
            n_hold_fixed += 1
        seen, dedup = set(), []
        for e in new_events:
            k = (e[0], e[1], e[2])
            if k not in seen:
                seen.add(k)
                dedup.append(e)
        ms = [e[2] for e in dedup if e[2] is not None]
        fixed_sus.append(dict(inst, events=sorted(dedup, key=lambda e: (e[0], e[1], e[2] or 0)), n=len(dedup),
                              range=[int(min(ms)), int(max(ms))] if ms else inst.get("range")))
    out = [i for i in out if i["kind"] != "sustain"] + fixed_sus
    holds = [1] * n_hold_fixed
    if progress:
        progress(f"{stem}: transcription fused - {n_chord} chord tones on hits, {len(holds)} held events re-voiced")
    return out, None


def merge_overlaps(events):
    """Within one voice, events of the SAME pitch that overlap in time are
    one note (their union), not two stacked: the transcription fusion
    re-voices each held beat with the notes sounding through it, which
    duplicated notes the reader already held long (a chord held 16 beats
    came out as the long notes PLUS a copy per beat - double energy and an
    attack every beat; found on the synthetic gate where the pads explained
    13-30% of their stem)."""
    by = {}
    for e in events:
        by.setdefault(e[2], []).append(list(e))
    out = []
    for m, evs in by.items():
        evs.sort(key=lambda e: (e[0] * STEPS + e[1], -e[3]))
        cur = None
        for e in evs:
            s, d = e[0] * STEPS + e[1], e[3]
            if cur is not None and s <= cur[0] * STEPS + cur[1] + cur[3]:
                end = max(cur[0] * STEPS + cur[1] + cur[3], s + d)
                cur[3] = int(end - (cur[0] * STEPS + cur[1]))
                cur[4] = round(max(cur[4], e[4]), 3)
                cur[5] = round(max(cur[5], e[5]), 3)
            else:
                if cur is not None:
                    out.append(cur)
                cur = e
        if cur is not None:
            out.append(cur)
    return sorted(out, key=lambda e: (e[0], e[1], e[2] if e[2] is not None else -1))


def _poly_share(events):
    by = {}
    for e in events:
        by[(e[0], e[1])] = by.get((e[0], e[1]), 0) + 1
    return sum(1 for v in by.values() if v > 1) / max(len(by), 1)


# --------------------------------------------------------------------------
# The pass
# --------------------------------------------------------------------------

READ_VOCALS = False            # read the vocals stem's notes (see identify); the program carries vocals as phrases either way
OTHER_VERIFY_OCTAVE = False    # every `other` note checked against the stem: the read pitch against the octave above and below
                               # by harmonic-summation salience over the note (see verify_other_notes). OFF: truth set
                               # 2026-09-09 - pop 0.29 -> 0.64 (the synth voices YourMT3 reads an octave off) but house 0.82
                               # -> 0.67, rock 0.64 -> 0.53, ballad 0.84 -> 0.73 (a dominant second harmonic reads as "the
                               # octave up" on pianos and guitars); median 0.73 -> 0.65. Per voice (voice_octave_shift) the
                               # same trade at 0.6 / 0.55. No salience rule tells a wrong octave from a bright instrument
OTHER_EXTEND_HOLDS = True      # ...and a note's end moved to where the stem stops holding its pitch (the transcriber ends holds
                               # early: rock organ 2-3 s written, 0.9 s read). KEPT: truth set 2026-09-09, alone - notes F1
                               # 0.73 / purity 0.72 unchanged, the voices' render against the true other stem 20.1 -> 17.3 dB
                               # (rock 25.1 -> 20.7, ballad 23.0 -> 17.2, pop 24.2 -> 23.0), level +3.8 -> +4.0
VERIFY_MARGIN_OCT = 1.3        # the octave must be this much more salient to replace the read pitch
EXTEND_KEEP = 0.5              # a note holds while its salience stays at this share of its own opening level...
EXTEND_MAX_S = 4.0             # ...at most this long
EXTEND_MIN_S = 0.25            # ...and only notes at least this long to begin with (a stab is a stab)
_SAL_FFT, _SAL_HOP = 4096, 1024


def salience_map(y, lo, hi, n_h=6):
    """(S [frames, midis lo..hi], hop_s): harmonic-summation salience of the stem per frame and semitone
    (the same measure the onset reader's pitch pick and the self-check use, over the whole stem at once)."""
    import librosa
    from scipy.ndimage import maximum_filter1d
    M = np.abs(librosa.stft(np.ascontiguousarray(y, dtype=np.float32), n_fft=_SAL_FFT, hop_length=_SAL_HOP))
    M = maximum_filter1d(M, 5, axis=0)
    df = RATE / float(_SAL_FFT)
    midis = np.arange(lo, hi + 1)
    f0 = 440.0 * 2 ** ((midis - 69) / 12.0)
    S = np.zeros((M.shape[1], len(midis)), dtype=np.float32)
    for h in range(1, n_h + 1):
        idx = np.round(h * f0 / df).astype(int)
        ok = idx < M.shape[0] - 3
        S[:, ok] += (M[idx[ok], :].T / (h ** 0.6)).astype(np.float32)
    return S, _SAL_HOP / RATE


def verify_other_notes(insts, y, beats, period, progress=None):
    """Two corrections on pitched `other` events, from the stem itself:
    octave: the read pitch is replaced by the octave above or below when that octave's salience over the
    note is VERIFY_MARGIN_OCT times the read one's; length: a note at least EXTEND_MIN_S long holds on while
    the stem's salience at its pitch stays above EXTEND_KEEP of the note's opening level, up to the voice's
    next onset at that pitch and EXTEND_MAX_S. -> (n_octave, n_extended)"""
    lo, hi = PITCH_RANGE.get("other", (36, 96))
    S, hop_s = salience_map(y, lo - 12, hi + 12)
    base = lo - 12
    n_oct = n_ext = 0
    step_s = period / STEPS
    for inst in insts:
        if not inst.get("pitched") or not inst.get("events"):
            continue
        evs = inst["events"]
        times = [event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0) for e in evs]
        by_pitch = {}
        for k, e in enumerate(evs):
            if e[2] is not None:
                by_pitch.setdefault(int(e[2]), []).append(times[k])
        for k, e in enumerate(evs):
            if e[2] is None:
                continue
            m = int(e[2])
            t = times[k]
            dur = max(0.08, e[3] * step_s)
            fa, fb = int((t + 0.02) / hop_s), int((t + min(dur, 0.6)) / hop_s) + 1
            if fb - fa < 1 or fb > len(S):
                continue
            if OTHER_VERIFY_OCTAVE:
                col = m - base
                s_here = float(np.median(S[fa:fb, col])) if 0 <= col < S.shape[1] else 0.0
                best_m, best_s = m, s_here
                for cand in (m - 12, m + 12):
                    c = cand - base
                    if lo <= cand <= hi and 0 <= c < S.shape[1]:
                        s_c = float(np.median(S[fa:fb, c]))
                        if s_c >= VERIFY_MARGIN_OCT * max(s_here, 1e-9) and s_c > best_s:
                            best_m, best_s = cand, s_c
                if best_m != m:
                    e[2] = best_m
                    m = best_m
                    n_oct += 1
            if OTHER_EXTEND_HOLDS and dur >= EXTEND_MIN_S:
                col = m - base
                if not (0 <= col < S.shape[1]):
                    continue
                f_on = int(t / hop_s)
                f_off = int((t + dur) / hop_s)
                if f_off <= f_on + 1 or f_off >= len(S):
                    continue
                opening = float(np.median(S[f_on + 1: min(f_off, f_on + 1 + max(2, int(0.3 / hop_s))), col]))
                if opening <= 0:
                    continue
                nxt = min([x for x in by_pitch.get(m, []) if x > t + 0.05], default=t + EXTEND_MAX_S + dur)
                f_max = min(len(S) - 1, int(min(nxt - 0.03, t + EXTEND_MAX_S) / hop_s))
                f = f_off
                while f < f_max and S[f, col] >= EXTEND_KEEP * opening:
                    f += 1
                new_dur = f * hop_s - t
                if new_dur > dur + 0.1:
                    e[3] = int(max(1, round(new_dur / step_s)))
                    n_ext += 1
        ms = [ev[2] for ev in evs if ev[2] is not None]
        if ms:
            inst["range"] = [int(min(ms)), int(max(ms))]
    if progress and (n_oct or n_ext):
        progress(f"other: {n_oct} notes moved an octave by the stem, {n_ext} holds extended to where the stem lets go")
    return n_oct, n_ext
OTHER_FAMILIES = False         # the `other` stem read per six-stem family (guitar / piano / rest, lib/dj/sixstem.py). OFF:
                               # truth set 2026-09-09 notes F1 0.60 -> 0.56 as built (notes leaking into two families read
                               # twice), 0.59 with the cross-family de-duplication; purity 0.47 -> 0.46 / 0.44 (house 0.61 ->
                               # 0.79, the rest flat or down); render gate `other` 11.9 -> 12.3 dB (organ 10 -> 24)
OTHER_FAMILY_FLOOR_DBFS = -50.0   # a family this quiet is not in the song...
OTHER_FAMILY_REL_DB = 18.0        # ...nor one this far under the stem (the separator's bleed of the other families)
OTHER_FAMILY_VOICES = {"guitar": 2, "piano": 2, "rest": 4}   # voices a family may split into
OTHER_FAMILY_DEDUPE = True     # the same note read in two families keeps the louder one (see _dedupe_families)
OTHER_DEDUPE_S = 0.03
YMT3_DEDUPE = True             # the same for YourMT3's program voices
YMT3_OCTAVE = False            # a program voice moved an octave when its notes' odd harmonics are missing (voice_octave_shift).
                               # OFF: at vote 0.6 / odd < 0.55 x even it rescued the pop track (0.32 -> 0.65) and broke house
                               # (0.83 -> 0.60) and the ballad (0.84 -> 0.74): piano voices with weak fundamentals; a stricter
                               # vote was measured separately (see the plan)
OCTAVE_VOTE = 0.6              # ...when this share of its sampled notes say so
OCTAVE_SAMPLE = 40
OCTAVE_UP_RATIO = 0.55         # odd harmonics under this share of the even ones = "the octave up" for one note
YMT3_OCTAVE_BY_READER = True   # a program voice moved an octave when the TRANSCRIPTION reader (basic-pitch, polyreader.read)
                               # hears its onsets at the other octave: per voice, its notes matched to that reading by onset
                               # (60 ms) at the same pitch (agree) or an octave up / down; a voice whose octave disagreements
                               # outnumber its agreements OCTAVE_READER_RATIO:1 (at least OCTAVE_READER_MIN of them) moves.
                               # Truth set 2026-09-10: pop 0.29 -> 0.68 (one synth voice, 255 of 256 notes an octave down),
                               # dnb 0.55 -> 0.59, the other four untouched; median 0.73 -> 0.75 = the same vote decided by
                               # the truth on every song. Where the salience vote above failed (a piano's second harmonic reads
                               # as "the octave up"), a second reader's opinion does not: it hears the note, not a harmonic
OCTAVE_READER_RATIO = 2.0
OCTAVE_READER_MIN = 10


def octave_vote_by_reader(insts, ref_notes, beats, period, progress=None):
    """Move whole `insts` voices by an octave where ref_notes [(t, midi)] (another reader of the same
    stem) hears their onsets an octave away. -> {voice id: shift} applied."""
    ref = {}
    for t, m in ref_notes:
        ref.setdefault(int(m), []).append(float(t))
    ref = {m: np.sort(np.array(ts)) for m, ts in ref.items()}

    def heard(t, m, tol=0.06):
        rt = ref.get(m)
        if rt is None:
            return False
        k = int(np.searchsorted(rt, t))
        return any(0 <= j < len(rt) and abs(rt[j] - t) <= tol for j in (k - 1, k))

    shifts = {}
    for inst in insts:
        if not inst.get("pitched"):
            continue
        agree = up = down = 0
        for e in inst["events"]:
            if e[2] is None:
                continue
            t = event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
            m = int(e[2])
            if heard(t, m):
                agree += 1
            else:
                up += heard(t, m + 12)
                down += heard(t, m - 12)
        shift = 0
        if up >= OCTAVE_READER_MIN and up >= OCTAVE_READER_RATIO * max(agree, down):
            shift = 12
        elif down >= OCTAVE_READER_MIN and down >= OCTAVE_READER_RATIO * max(agree, up):
            shift = -12
        if not shift:
            continue
        for e in inst["events"]:
            if e[2] is not None:
                e[2] = int(e[2]) + shift
        ms = [e[2] for e in inst["events"] if e[2] is not None]
        inst["range"] = [int(min(ms)), int(max(ms))]
        if inst.get("exemplar_midi") is not None:
            inst["exemplar_midi"] = int(inst["exemplar_midi"]) + shift
        shifts[inst.get("id") or inst.get("hint") or len(shifts)] = shift       # ids are given after the stem is read
        if progress:
            progress(f"other: {inst.get('hint')} voice moved {shift:+d} semitones (the transcription reader hears it there: "
                     f"{max(up, down)} notes against {agree})")
    return shifts


def voice_octave_shift(inst, y, beats, period, lo=None, hi=None):
    """+12, -12 or 0 for a pitched voice as a whole: over up to OCTAVE_SAMPLE loud notes, the stem's spectrum
    at the read pitch's odd harmonics (1, 3, 5) against its even ones (2, 4, 6). Odd missing -> the note is the
    octave UP (the read pitch was the sub-harmonic); the octave below's odd harmonics (f0/2, 3f0/2, 5f0/2)
    present at half the read pitch's first three -> the octave DOWN."""
    from scipy.ndimage import maximum_filter1d
    lo_r, hi_r = PITCH_RANGE.get(inst["stem"], (24, 108))
    lo, hi = lo or lo_r, hi or hi_r
    evs = [e for e in inst["events"] if e[2] is not None]
    if len(evs) < 6:
        return 0
    evs = sorted(evs, key=lambda e: -e[4])[:OCTAVE_SAMPLE]
    up = down = n = 0
    for e in evs:
        t = event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
        a, b = int((t + 0.02) * RATE), int(min(len(y), (t + min(0.4, max(0.12, e[3] * period / STEPS))) * RATE))
        if b - a < 1024:
            continue
        R, df = _spectrum(y[a:b], _N_HPS)
        Rm = maximum_filter1d(R, 5)
        pk = lambda f: float(Rm[int(round(f / df))]) if 0 < int(round(f / df)) < len(Rm) else 0.0
        f0 = 440.0 * 2 ** ((int(e[2]) - 69) / 12.0)
        odd = pk(f0) + pk(3 * f0) + pk(5 * f0)
        even = pk(2 * f0) + pk(4 * f0) + pk(6 * f0)
        below = pk(f0 / 2) + pk(1.5 * f0) + pk(2.5 * f0)
        n += 1
        if odd < OCTAVE_UP_RATIO * even and int(e[2]) + 12 <= hi:
            up += 1
        elif below >= 0.5 * (pk(f0) + pk(2 * f0) + pk(3 * f0)) and int(e[2]) - 12 >= lo:
            down += 1
    if n < 6:
        return 0
    if up / n >= OCTAVE_VOTE:
        return 12
    if down / n >= OCTAVE_VOTE:
        return -12
    return 0


def _dedupe_families(insts, beats):
    """Across `other` families: events of the same pitch within OTHER_DEDUPE_S of one another are one
    note; it stays with the voice whose event is louder (its velocity against its own loud hits plus the
    voice's level). -> number of events removed."""
    entries = []
    for vi, inst in enumerate(insts):
        for ei, e in enumerate(inst["events"]):
            if e[2] is None:
                continue
            t = event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
            loud = float(inst.get("level_dbfs") or 0.0) + (float(e[4]) - 1.0) * 24.0
            entries.append((int(e[2]), t, loud, vi, ei))
    entries.sort()
    drop = {vi: set() for vi in range(len(insts))}
    i = 0
    while i < len(entries):
        j = i + 1
        group = [entries[i]]
        while j < len(entries) and entries[j][0] == entries[i][0] and entries[j][1] - group[-1][1] <= OTHER_DEDUPE_S:
            group.append(entries[j])
            j += 1
        fams = {g[3] for g in group}
        if len(fams) > 1:
            keep = max(group, key=lambda g: g[2])
            for g in group:
                if g[3] != keep[3]:
                    drop[g[3]].add(g[4])
        i = j
    n = 0
    for vi, inst in enumerate(insts):
        if drop[vi]:
            inst["events"] = [e for k, e in enumerate(inst["events"]) if k not in drop[vi]]
            inst["n"] = len(inst["events"])
            n += len(drop[vi])
    return n
PARALLEL_READERS = 3           # stem readers run in this many threads (1 = one after another)


def _read_stem(name, y, beats, period, progress, drums_y):
    """One stem's reading -> (record, reasons). progress is thread-safe."""
    rec = {"n_onsets": 0, "instruments": []}
    reasons = []
    peak = float(np.abs(y).max()) if len(y) else 0.0
    if peak < 1e-3 or len(y) < 4 * RATE:
        reasons.append(f"{name}: silent stem")
        return rec, reasons
    if name == "bass" and drums_y is not None:
        y, g = clean_bass(y, drums_y)
        rec["kick_bleed_gain"] = round(g, 3)
        if progress:
            progress(f"bass: kick bleed subtracted (gain {g:.2f})")
    if progress:
        progress(f"{name}: onsets")
    insts = []
    n_on = 0
    if name == "drums":
        try:
            insts, n_on = _drum_instruments(y, beats, period, progress=progress)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{name}: drum identification failed ({type(e).__name__}: {str(e)[:80]})")
    elif name == "other" and OTHER_READER == "ymt3":
        # notes labelled with an instrument by YourMT3+ (lib/dj/ymt3.py, its own interpreter); falls back to the
        # basic-pitch reader when the model is not available
        try:
            from lib.dj import ymt3 as YM, polyreader as PR
            labelled = YM.transcribe(y, progress=progress)
            if labelled:
                insts = PR.read_labelled(y, beats, period, labelled, stem=name, progress=progress)
                if YMT3_DEDUPE and len(insts) > 1:
                    # the same note under two programs (truth set dnb: 853 notes for 496 written, precision 0.43)
                    n_dropped = _dedupe_families(insts, beats)
                    insts = [i for i in insts if i["n"] >= 4]
                    if progress and n_dropped:
                        progress(f"other: {n_dropped} notes duplicated across program voices dropped")
                if YMT3_OCTAVE:
                    # a voice read an octave off as a whole (truth set pop: synth brass and lead, F1 0.31 exact
                    # against 0.63 octave-blind): the stem's odd harmonics decide per voice
                    for inst in insts:
                        shift = voice_octave_shift(inst, y, beats, period)
                        if shift:
                            for e in inst["events"]:
                                if e[2] is not None:
                                    e[2] = int(e[2]) + shift
                            ms = [e[2] for e in inst["events"] if e[2] is not None]
                            inst["range"] = [int(min(ms)), int(max(ms))]
                            inst["exemplar_midi"] = int(inst.get("exemplar_midi") or ms[0]) + shift
                            if progress:
                                progress(f"other: {inst.get('hint')} voice moved {shift:+d} semitones (its odd harmonics were missing)")
                if YMT3_OCTAVE_BY_READER and insts:
                    # the transcription reader's notes as a second opinion on each voice's octave (see the flag)
                    ref = PR.read(y, beats, period, stem=name, progress=None)
                    ref_notes = [(event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0), int(e[2]))
                                 for i in ref for e in i["events"] if e[2] is not None]
                    octave_vote_by_reader(insts, ref_notes, beats, period, progress=progress)
            else:
                insts = PR.read(y, beats, period, stem=name, progress=progress)
            if OTHER_VERIFY_OCTAVE or OTHER_EXTEND_HOLDS:
                verify_other_notes(insts, y, beats, period, progress=progress)
            n_on = sum(i["n"] for i in insts)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{name}: YourMT3 reading failed ({type(e).__name__}: {str(e)[:80]})")
    elif name == "other" and OTHER_READER == "transcription":
        # the polyphonic stem from a transcription, voices by timbre (lib/dj/polyreader.py): on the truth
        # set (demucs stems) notes F1 0.29 -> 0.60 median over the onset-first reader
        try:
            from lib.dj import polyreader as PR
            if OTHER_FAMILIES:
                # guitar / piano / the rest, each read alone (lib/dj/sixstem.py): the parts that shared one
                # voice - the reconstruction's "wrong instrument on a part" - are separated by a model that
                # was trained on exactly those instruments, where no clustering rule could
                from lib.dj import sixstem as SX
                subs = SX.separate_other(y, progress=progress)
                insts = []
                stem_lvl = 20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-9)
                for fam in SX.FAMILIES:
                    y_sub = subs.get(fam)
                    if y_sub is None:
                        continue
                    lvl = 20 * np.log10(np.sqrt(np.mean(y_sub.astype(np.float64) ** 2)) + 1e-9)
                    if lvl < OTHER_FAMILY_FLOOR_DBFS or lvl < stem_lvl - OTHER_FAMILY_REL_DB:
                        continue                         # not in the song, or the model's bleed (house: guitar 20 dB under)
                    fam_insts = PR.read(y_sub, beats, period, stem=name, progress=progress, max_voices=OTHER_FAMILY_VOICES.get(fam))
                    for inst in fam_insts:
                        inst["family"] = fam
                        if fam in ("guitar", "piano"):
                            inst["hint"] = fam
                    insts.extend(fam_insts)
                if OTHER_FAMILY_DEDUPE and len(insts) > 1:
                    # a note the separator leaks into two families is transcribed twice (truth set: dnb 597 -> 924
                    # notes for 496 written): the same pitch within OTHER_DEDUPE_S stays with the louder family
                    n_dropped = _dedupe_families(insts, beats)
                    insts = [i for i in insts if i["n"] >= 4]
                    if progress and n_dropped:
                        progress(f"other: {n_dropped} notes duplicated across families dropped")
                if progress:
                    progress("other: families " + ", ".join(f"{f} {sum(1 for i in insts if i.get('family') == f)}" for f in SX.FAMILIES))
                # levels relative across the families (each family's reader set them within itself)
                if insts:
                    top = max(i["level_dbfs"] for i in insts)
                    for inst in insts:
                        inst["level_db"] = round(float(inst["level_dbfs"] - top), 1)
                insts.sort(key=lambda d: -d["n"])
            else:
                insts = PR.read(y, beats, period, stem=name, progress=progress)
            n_on = sum(i["n"] for i in insts)
            if SUSTAIN_FUSION:
                # the transcriber ends held notes early and misses organ / pad holds (truth set: organ
                # recall 0.18); the reader's own held-note pass fills the beats no transcribed note covers
                H, _P = _hpss(y)
                insts += _fuse_holds(y, H, beats, period, name, insts, progress=progress)
                del H, _P
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{name}: transcription reading failed ({type(e).__name__}: {str(e)[:80]})")
    elif name == "bass" and BASS_READER == "pitchtrack":
        # the bass as one monophonic line from a pitch track (lib/dj/bassreader.py): on the truth set
        # (tools/tests/_dj_truthset.py, demucs stems) notes F1 0.61 -> 0.78 median over the onset-first reader
        try:
            from lib.dj import bassreader as BR
            insts = BR.read(y, beats, period, progress=progress)
            n_on = sum(i["n"] for i in insts)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{name}: pitch-track reading failed ({type(e).__name__}: {str(e)[:80]})")
    else:
        H, P = _hpss(y)
        try:
            # bass: onsets on the whole (cleaned) signal - a bass note's attack is mostly harmonic and
            # the percussive part misses or mistimes it (measured 2026-09-09: chroma 0.30 -> 0.69 on
            # Evolution, up on all four tracks); plucks and voices keep the percussive part
            insts, n_on = _pluck_instruments(y, y if name == "bass" else P, beats, period, name, progress=progress)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{name}: hit identification failed ({type(e).__name__}: {str(e)[:80]})")
        try:
            insts += _sustain_instruments(y, H, beats, period, name, hit_insts=insts, progress=progress)
        except Exception as e:  # noqa: BLE001
            reasons.append(f"{name}: sustain reading failed ({type(e).__name__}: {str(e)[:80]})")
        del H, P
        if name in VERIFY_STEMS:
            lo_v, hi_v = PITCH_RANGE[name]
            n_changed = 0
            for inst in insts:
                if inst["kind"] == "hit" and inst.get("pitched"):
                    inst["events"], n = verify_pitches(y, inst["events"], beats, period, lo_v, hi_v)
                    n_changed += n
                    ms = [e[2] for e in inst["events"] if e[2] is not None]
                    if ms:
                        inst["range"] = [int(min(ms)), int(max(ms))]
            if progress:
                progress(f"{name}: {n_changed} pitches corrected over the whole note")
        if name in POLY_STEMS:
            try:
                insts, why = _poly_refine(y, insts, beats, period, name, progress=progress)
                if why:
                    reasons.append(why)
            except Exception as e:  # noqa: BLE001
                reasons.append(f"{name}: polyphonic refinement failed ({type(e).__name__}: {str(e)[:80]})")
    rec["n_onsets"] = int(n_on)
    for i, inst in enumerate(insts):
        inst["id"] = f"{name}.{i + 1}"
        if inst["kind"] == "sustain":
            inst["events"] = merge_overlaps(inst["events"])
            inst["n"] = len(inst["events"])
        inst["label"], inst["detail"] = describe(inst)
    rec["instruments"] = insts
    if progress:
        progress(f"{name}: {len(insts)} instruments")
    return rec, reasons


def identify(stems, beats, down0=0, period=None, progress=None, stem_names=STEM_ORDER, explain=True):
    """stems: {name: (n,2) or (n,) float array @44100} (any dtype);
    beats: beat times (s) from beat_times(). -> the result dict.
    explain: afterwards every instrument is played back against its stem
    (lib/dj/explain.py): its record gets `explained` (the share of the
    stem it accounts for in its own windows), voices that are not there
    are dropped and voices that are one sound joined; result["pruned"]
    says what went and why."""
    beats = np.asarray(beats, dtype=np.float64)
    if len(beats) < 8:
        raise ValueError("no usable beat grid for this track")
    period = float(period or np.median(np.diff(beats)))
    result = {"version": INSTRUMENTS_VERSION, "steps": STEPS, "period_s": round(period, 5),
              "beats": [round(float(t), 4) for t in beats], "down0": int(down0), "stems": {}, "reasons": []}
    names = [n for n in stem_names if n in stems]
    if not READ_VOCALS and "vocals" in names:
        # vocals are carried as reused phrases of the recording (lib/dj/voices.phrase_library, from the stem
        # audio, not from notes); their note reading (the old onset + sustain pass, 30-60 s a track) only fed
        # display rows nothing plays. Off by default since 2026-09-09
        names.remove("vocals")
        result["stems"]["vocals"] = {"n_onsets": 0, "instruments": []}
        result["reasons"].append("vocals: carried as phrases (note reading off)")
    import threading
    lock = threading.Lock()

    def say(msg):
        if progress:
            with lock:
                progress(msg)
    audio = {}

    def load(name):
        # a stem may be given as a callable (decode on demand)
        if name not in audio:
            audio[name] = _mono(stems[name]() if callable(stems[name]) else stems[name])
        return audio[name]
    outcomes = {}
    if PARALLEL_READERS > 1 and len(names) > 1:
        # the stems' readers are independent (the bass waits for the drums: its kick bleed is subtracted with
        # them) and mostly C-level work (torch, onnx, numba, numpy release the GIL): threads overlap them.
        # Measured 2026-09-09: a reading is the SUM of four readers; this makes it the slowest one
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=PARALLEL_READERS) as pool:
            futs = {}
            # readers that need the GPU on their own go after the drums (drumsep holds the card while it runs):
            # the bass (its kick bleed is subtracted with the drums) and YourMT3 (another process on the same card)
            after_drums = {"bass"} | ({"other"} if OTHER_READER == "ymt3" else set())
            if "drums" in names:
                futs["drums"] = pool.submit(_read_stem, "drums", load("drums"), beats, period, say, None)
            for name in names:
                if name == "drums" or name in after_drums:
                    continue
                futs[name] = pool.submit(_read_stem, name, load(name), beats, period, say, None)
            drums_y = None
            if "drums" in futs and any(n in names for n in after_drums):
                futs["drums"].result()
                drums_y = audio.get("drums")
            if "bass" in names:
                futs["bass"] = pool.submit(_read_stem, "bass", load("bass"), beats, period, say, drums_y)
            if "other" in names and "other" in after_drums:
                futs["other"] = pool.submit(_read_stem, "other", load("other"), beats, period, say, None)
            for name in names:
                outcomes[name] = futs[name].result()
    else:
        drums_y = None
        for name in names:
            y = load(name)
            if name == "drums":
                drums_y = y
            outcomes[name] = _read_stem(name, y, beats, period, say, drums_y if name == "bass" else None)
    for name in names:
        rec, reasons = outcomes[name]
        result["stems"][name] = rec
        result["reasons"].extend(reasons)
    audio.clear()
    if COMPLETE_BASS and "bass" in result["stems"] and "other" in result["stems"]:
        n_moved = complete_bass(result, beats, period)
        if progress and n_moved:
            progress(f"bass: {n_moved} notes recovered from the other stem")
    if explain:
        from lib.dj import explain as EX
        result = EX.explain_pass(result, stems, progress=progress)
    return result


COMPLETE_BASS = False          # short single notes the separator put into `other` that sit in gaps of the bass line, an
                               # octave or less from both neighbours, move to the bass voice (complete_bass). Truth set:
                               # demucs puts every one of the pop track's 1-step octave pops (A2-A3) into `other` (bass stem
                               # -42 dBFS during the pop, `other` -25, true stem -27) - 44 of 44 missed from the bass stem alone.
                               # OFF: measured 2026-09-09 - 26 notes moved on that track, none of them a pop (bass F1 0.68 ->
                               # 0.66), 2 wrong moves on rock, nothing elsewhere. The pop sits on the pad's root pitch, so the
                               # transcriber reads it as the pad continuing; there is no note in `other` to move
COMPLETE_MAX_S = 0.45          # ...a candidate is at most this long
COMPLETE_NEAR_S = 0.6          # ...and has a bass note starting within this on at least one side
COMPLETE_ABOVE = 12            # ...at most this far above the bass line's highest read note
COMPLETE_GAP_S = 0.03          # a candidate must start in a gap of the line (no bass note sounding beyond this tolerance)


def complete_bass(result, beats, period):
    """Move `other` notes that are bass notes by every structural test into
    the bass voice: monophonic in their voice (no other note at the same
    beat and step), short, within the bass range and no more than
    COMPLETE_ABOVE above the line's highest note, starting in a gap of the
    line, an octave or less from the bass notes on both sides and within
    COMPLETE_NEAR_S of one of them. -> number moved. No spectral test is
    possible: the note is NOT in the bass stem, which is the whole point."""
    bass_insts = [i for i in result["stems"]["bass"]["instruments"] if i["kind"] == "hit" and i.get("pitched")]
    other_insts = [i for i in result["stems"]["other"]["instruments"] if i["kind"] == "hit" and i.get("pitched")]
    if not bass_insts or not other_insts:
        return 0
    bass = max(bass_insts, key=lambda i: i["n"])
    step_s = period / STEPS
    line = []
    for e in bass["events"]:
        if e[2] is None:
            continue
        t = event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
        line.append((t, t + max(1, e[3]) * step_s, int(e[2])))
    if len(line) < 8:
        return 0
    line.sort()
    starts = np.array([a for a, _b, _m in line])
    ends = np.array([b for _a, b, _m in line])
    pitches = np.array([m for _a, _b, m in line])
    lo, hi = PITCH_RANGE["bass"]
    top = min(hi, int(pitches.max()) + COMPLETE_ABOVE)
    moved = 0
    new_events = []
    for inst in other_insts:
        keep = []
        at = {}
        for e in inst["events"]:
            at.setdefault((e[0], e[1]), 0)
            at[(e[0], e[1])] += 1
        for e in inst["events"]:
            m = e[2]
            ok = m is not None and lo <= m <= top and at[(e[0], e[1])] == 1 and e[3] * step_s <= COMPLETE_MAX_S
            if ok:
                t = event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
                k = int(np.searchsorted(starts, t))
                prev_i, next_i = k - 1, k
                sounding = prev_i >= 0 and ends[prev_i] > t + COMPLETE_GAP_S
                near = (prev_i >= 0 and t - starts[prev_i] <= COMPLETE_NEAR_S) or (next_i < len(starts) and starts[next_i] - t <= COMPLETE_NEAR_S)
                close = all(abs(int(pitches[j]) - m) <= 12 for j in (prev_i, next_i) if 0 <= j < len(starts))
                ok = (not sounding) and near and close and (prev_i >= 0 or next_i < len(starts))
            if ok:
                new_events.append(list(e))
                moved += 1
            else:
                keep.append(e)
        if len(keep) != len(inst["events"]):
            inst["events"] = keep
            inst["n"] = len(keep)
    if new_events:
        bass["events"] = sorted(bass["events"] + new_events, key=lambda e: (e[0], e[1], e[2] or 0))
        bass["n"] = len(bass["events"])
        ms = [e[2] for e in bass["events"] if e[2] is not None]
        bass["range"] = [int(min(ms)), int(max(ms))]
        bass["recovered"] = moved
        result["stems"]["other"]["instruments"] = [i for i in result["stems"]["other"]["instruments"] if i["n"] >= 4 or i["kind"] != "hit"]
    return moved


def identify_track(music_root, track_id, grid, downbeat_offset, duration_s, bpm=None, progress=None, save_result=True):
    """Run the pass on the track's stems from disk (decoded one at a time,
    mono - a five-minute track's four stereo stems would be 400 MB
    before the analysis makes its own copies), store the result."""
    from lib.dj.stems import stem_paths
    beats, down0 = beat_times(grid, downbeat_offset, duration_s, bpm=bpm)
    if len(beats) < 8:
        raise ValueError("no beat grid")
    paths = stem_paths(music_root, track_id)
    if paths is None:
        raise FileNotFoundError("no stems on disk")

    def loader(p):
        def decode():
            from lib.dj.features import decode_file_stereo
            return _mono(decode_file_stereo(p))
        return decode
    res = identify({n: loader(p) for n, p in paths.items()}, beats, down0=down0, progress=progress)
    if save_result:
        save(music_root, track_id, res)
    return res


# --------------------------------------------------------------------------
# Reading the result
# --------------------------------------------------------------------------

def instruments(result):
    """All instruments in stem order."""
    out = []
    for name in STEM_ORDER:
        out.extend((result.get("stems") or {}).get(name, {}).get("instruments") or [])
    return out


def bar_beat(result, k):
    """(bar number, beat in bar 1..4) of beat index k, bars counted from the first downbeat."""
    d = int(result.get("down0", 0))
    return (k - d) // 4, (k - d) % 4 + 1


def beat_index(result, t):
    beats = result.get("beats") or []
    if not beats:
        return None
    pos = beat_step_of(np.asarray(beats), t, result.get("period_s"))
    return pos[0] if pos else None


def beat_table(result):
    """beat index -> [(instrument, event)] for every event that STARTS
    on or SPANS that beat (sustained notes span). Built once per result."""
    table = {}
    for inst in instruments(result):
        for ev in inst["events"]:
            b, st, midi, dur, vel, conf = ev[:6]
            n_beats = max(1, int(np.ceil((st + dur) / STEPS)))
            for k in range(b, b + n_beats):
                table.setdefault(k, []).append((inst, ev))
    return table


def display_name(inst):
    """The name a row shows: the role read for the sound when there is
    one (kick, pad, stab...), else its measured label."""
    hint = inst.get("hint")
    if hint:
        return hint
    return inst.get("label") or inst.get("id", "?")


def facts(inst):
    """The measured facts, short: register · length · hits · level · range."""
    parts = [_register(inst.get("centroid_hz", 1000.0))]
    if inst["kind"] == "sustain":
        parts.append("held")
    else:
        parts.append(f"{1000 * inst.get('dur_s', 0):.0f} ms")
    parts.append(f"{inst.get('n', 0)} {'beats' if inst['kind'] == 'sustain' else 'hits'}")
    if inst.get("level_db") is not None:
        parts.append(f"{inst['level_db']:+.0f} dB")
    if inst.get("range"):
        lo, hi = inst["range"]
        parts.append(f"{note_name(lo)}-{note_name(hi)}" if lo != hi else note_name(lo))
    return " · ".join(parts)


def describe_beat(result, table, k):
    """One readable line for beat k, grouped by stem:
    'bar 24.2 | drums: kick, hat, hat+2 | bass: bass E2 | other: pad A2 E3'."""
    if k is None:
        return ""
    bar, bib = bar_beat(result, k)
    by_stem = {}
    for inst, ev in table.get(k, []):
        step = f"+{ev[1]}" if ev[1] else ""
        note = f" {note_name(ev[2])}" if ev[2] is not None else ""
        name = display_name(inst)
        if inst["kind"] == "sustain":
            by_stem.setdefault(inst["stem"], {}).setdefault(name, []).append(note.strip())
        else:
            by_stem.setdefault(inst["stem"], {}).setdefault(name + step, []).append(note.strip())
    parts = [f"bar {bar}.{bib}"]
    for stem in STEM_ORDER:
        if stem not in (result.get("stems") or {}):
            continue
        items = by_stem.get(stem) or {}
        if not items:
            parts.append(f"{stem}: -")
            continue
        words = []
        for name, notes in items.items():
            notes = [n for n in notes if n]
            words.append(name + (" " + " ".join(dict.fromkeys(notes)) if notes else ""))
        parts.append(f"{stem}: " + ", ".join(words))
    return "  |  ".join(parts)


def summary(result):
    """One line per instrument."""
    lines = []
    for inst in instruments(result):
        hint = f" ~{inst['hint']}" if inst.get("hint") else ""
        ex = f"  explains {100 * inst['explained']:.0f}%" if inst.get("explained") is not None else ""
        if inst.get("overshoot", 0) > 0.5:
            ex += f" (overshoots {100 * inst['overshoot']:.0f}%)"
        lines.append(f"{inst['id']:10s} {inst['label']:22s}{hint:12s} {inst['detail']}{ex}")
    for g in result.get("pruned") or []:
        if g.get("kind") == "gain":
            lines.append(f"{g['id']:10s} level moved {g.get('db', 0):+.0f} dB ({g['name']}): {g['why']}" + (f" [{g['gap']}]" if g.get("gap") else ""))
        else:
            lines.append(f"{g['id']:10s} dropped ({g['name']}, {g['n']} events): {g['why']}" + (f" [{g['gap']}]" if g.get("gap") else ""))
    return lines
