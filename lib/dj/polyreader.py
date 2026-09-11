"""The `other` stem read from a polyphonic TRANSCRIPTION, with the song's
voices found by clustering the transcribed notes' timbre - instead of
onset clusters first and a pitch guess per onset.

Why (2026-09-09, the truth set): on demucs `other` stems the onset-first
reader scores notes F1 0.29 (median of four songs) while basic-pitch
alone scores 0.60; on real songs the reader agrees with basic-pitch at
0.14-0.34. So the notes come from the transcriber, and what the reader
is good at - telling the song's sounds apart by timbre - assigns every
note to a voice.

    read(y, beats, period, stem="other") -> [instrument] in the reading's format

Notes starting together (within CHORD_S) are one GROUP (a chord, a stab):
the group is the unit of timbre (mel spectrum of its first 100 ms, its
length, register and how many notes it has) and of voice assignment, so
a chord's tones land in one voice. A voice whose groups are mostly long
is a held voice ("sustain", additive in the program), else a struck one.
"""
import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
STEPS = INS.STEPS
AMP_MIN = 0.3                  # basic-pitch amplitude below this is not a note
MIN_NOTE_S = 0.05
CHORD_S = 0.025                # notes starting within this are one group
FRAG_GAP_S = 0.0               # same-pitch transcribed notes this close are one note (0 = OFF: measured on the truth set
                               # 2026-09-09, 0.1 s took the median F1 0.60 -> 0.38 - repeated stabs and clav chops ARE
                               # separate notes; the transcriber's re-triggers on a held pad are the smaller error)
MAX_VOICES = {"other": 5, "vocals": 2, "bass": 2}
HELD_S = 0.45                  # a group longer than this is held; a voice is "sustain" when most of its groups are
ATTACK_S = 0.1
VEL_RANGE_DB = 24.0
FEATURES = "mel"               # "mel": attack/sustain mel bands | "profile": harmonic profile + hold change (dB units) with
                               # co-occurrence merging - measured 2026-09-09 on the truth set: purity 0.29/0.70/0.33/0.46
                               # against mel's 0.39/0.56/0.33/0.46; neither separates the GM parts in a demucs stem, the
                               # note-to-voice assignment is still open
MERGE_PROFILE_DB = 9.0         # clusters with profiles this close (dB rms) and...
MERGE_OVERLAP = 0.05           # ...notes overlapping in under this share of cases are one instrument
CLUSTER_STANDARDIZE = True     # z-score the group features before clustering ("mel" features)
CLUSTER_MIN_SIL = 0.25         # silhouette a split needs (the reader's default 0.35 kept every song at one voice)
CLUSTER = "kmeans_sil"         # "kmeans_sil": k-means, k by silhouette (INS._cluster_auto) | "gmm_bic": Gaussian mixture, k by
                               # BIC | "agglo": Ward agglomerative under a distance threshold (see _cluster_groups). Measured
                               # 2026-09-09: reader harness F1 0.62 / purity 0.46 (k-means), 0.60 / 0.44 (gmm), 0.60-0.61 /
                               # 0.48 (agglo; house purity 0.60 -> 0.71, funk 0.46 -> 0.56); the RENDER GATE (true notes through
                               # the voices vs the true parts) said no to agglo: `other` 11.9 -> 14.3 dB - the pop pad gained
                               # a voice of its own (18.5 -> 14.3), the house pad, pluck and dnb keys lost, the funk horns
                               # stayed at 35 dB in a voice of their own
AGGLO_THRESHOLD = 12.0         # agglo: Ward distance (standardised features) under which two clusters join
GMM_BIC_MARGIN = 0.02          # gmm_bic: a larger k needs this relative BIC improvement over the best smaller one
CLUSTER_MIN_SHARE = 0.05       # a voice needs this share of the groups (smaller ones fold into the nearest)


def _fold_small(X, labels, min_share):
    n = len(labels)
    ids = sorted(set(labels.tolist()))
    small = [i for i in ids if (labels == i).sum() < max(4, min_share * n)]
    keep = [i for i in ids if i not in small]
    if small and keep:
        cents = {i: X[labels == i].mean(axis=0) for i in keep}
        for j in np.where(np.isin(labels, small))[0]:
            labels[j] = min(keep, key=lambda i: float(np.linalg.norm(X[j] - cents[i])))
    # relabel 0..k-1
    remap = {l: i for i, l in enumerate(sorted(set(labels.tolist())))}
    return np.array([remap[l] for l in labels], dtype=int)


def _cluster_groups(F, k_max, stem):
    """Group features -> voice labels by the CLUSTER method."""
    n = len(F)
    if n < 16:
        return np.zeros(n, dtype=int)
    if CLUSTER == "kmeans_sil":
        return INS._cluster_auto(F, k_max, min_share=CLUSTER_MIN_SHARE, min_sil=CLUSTER_MIN_SIL, standardize=CLUSTER_STANDARDIZE)
    X = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-6) if CLUSTER_STANDARDIZE else np.asarray(F, dtype=np.float64)
    if CLUSTER == "gmm_bic":
        from sklearn.mixture import GaussianMixture
        best, best_bic, labels = None, None, np.zeros(n, dtype=int)
        for k in range(1, int(min(k_max, n // 8)) + 1):
            gm = GaussianMixture(k, covariance_type="diag", n_init=2, random_state=0, reg_covar=1e-3).fit(X)
            bic = float(gm.bic(X))
            if best_bic is None or bic < best_bic - GMM_BIC_MARGIN * abs(best_bic):
                best, best_bic = gm, bic
        labels = best.predict(X) if best is not None else labels
        return _fold_small(X, labels.astype(int), CLUSTER_MIN_SHARE)
    if CLUSTER == "agglo":
        from sklearn.cluster import AgglomerativeClustering
        ac = AgglomerativeClustering(n_clusters=None, distance_threshold=AGGLO_THRESHOLD, linkage="ward").fit(X)
        labels = ac.labels_.astype(int)
        if len(set(labels.tolist())) > k_max:
            ac = AgglomerativeClustering(n_clusters=k_max, linkage="ward").fit(X)
            labels = ac.labels_.astype(int)
        return _fold_small(X, labels, CLUSTER_MIN_SHARE)
    raise ValueError(CLUSTER)
SUSTAIN_FROM_S = 0.15          # the sustained spectrum is read from here...
SUSTAIN_TO_S = 0.4             # ...to here after the onset (held vs struck tells voices apart)
CONT_MERGE = True              # continuation merge (see merge_continuations): a same-pitch note the transcriber re-triggered
                               # inside a hold is folded into the note it continues when there is NO attack there
CONT_GAP_S = 0.3               # ...the earlier same-pitch note ended within this (or still sounds)
CONT_SCOPE = "voice"           # "voice": continuations fold only into a predecessor of the SAME voice (after clustering) |
                               # "all": into any same-pitch predecessor (before clustering)
CONT_TEST = "spectral"         # "spectral": the pitch's harmonic energy rises under CONT_RISE_DB | "onset": the transcriber's
                               # own onset posterior at the note's pitch and start is under CONT_ONSET_MIN (measured 2026-09-09
                               # on the truth set: WORSE, median 0.60 -> 0.52 - basic-pitch makes repeated stabs from
                               # activation jumps, not from its onset head, so their posterior is as low as a hold's)
CONT_ONSET_MIN = 0.3
CONT_ONSET_WIN_S = 0.035       # ...within this of the start, at the pitch and its neighbours (+-1 semitone)
CONT_RISE_DB = 8.0             # a continuation: the pitch's harmonic energy rises LESS than this from its lowest point in the
                               # 50 ms before the start to its peak in the 80 ms after...
CONT_DIP_DB = 4.0              # ...and dipped less than this against its level 70-120 ms before (a re-strike falls, then rises)...
CONT_PRED_S = 0.2              # ...and the note it would continue is at least this long (a hold's fragment chain begins with
                               # a long note; a 16th-note clav chop re-strikes its predecessor before it has decayed and its
                               # rise is small too - the predecessor's length is what tells them apart)
CONT_BEFORE_S, CONT_AFTER_S, CONT_DIP_S = 0.12, 0.08, 0.05
CONT_N_FFT, CONT_HOP = 1024, 128   # 23 ms windows every 2.9 ms: the dip between a re-struck note and its predecessor is
                               # 30 ms wide on 16ths; longer windows (46, 93 ms) smear it away
HOLD_BACKTRACK = False         # a note that absorbed continuations (a hold) starts where its pitch's harmonic energy rose,
                               # not where the transcriber's first fragment began (truth set: a pad's first fragment starts a
                               # median 0.5 s after the true onset - pad recall 0.11, strings 0.28, organ 0.19). OFF: measured
                               # 2026-09-09 on the truth set median 0.61 -> 0.60 (pad recall 0.08 -> 0.06, house 0.79 -> 0.76),
                               # and per voice (CONT_SCOPE "voice") 0.62 -> 0.61: the pitch's energy is shared with other
                               # parts' notes, and the walk back lands on them
HOLD_BACK_MAX_S = 2.0          # ...searched this far back...
HOLD_BACK_DB = 6.0             # ...while the energy stays within this of the fragment's level; the onset is the last rise of
                               # HOLD_BACK_DB within HOLD_BACK_RISE_S before the search stops
HOLD_BACK_RISE_S = 0.06
GHOST_DROP = True              # sub-octave ghosts (see drop_ghosts): a transcribed note whose odd harmonics are missing
GHOST_ODD_DB = -12.0           # ...odd harmonics this far under the even ones over the note is the octave below a real note


def _harmonic_frames(y):
    """STFT magnitude (dB) up to 6 kHz at CONT_HOP -> (S_db [frames, bins], df, hop_s)."""
    import librosa
    S = np.abs(librosa.stft(np.ascontiguousarray(y, dtype=np.float32), n_fft=CONT_N_FFT, hop_length=CONT_HOP, center=True))
    df = RATE / float(CONT_N_FFT)
    n_bins = int(6000.0 / df)
    S = S[:n_bins].T.astype(np.float32)
    return 20.0 * np.log10(S + 1e-6), df, CONT_HOP / RATE


def _pitch_energy_db(S_db, df, m, f0_frame, f1_frame, harmonics=(1, 2, 3, 4, 5, 6)):
    """Per frame in [f0_frame, f1_frame): the summed power at the harmonics of midi m (max over +-1 bin), in dB."""
    f0 = 440.0 * 2 ** ((m - 69) / 12.0)
    seg = S_db[max(0, f0_frame):max(0, f1_frame)]
    if len(seg) == 0:
        return np.zeros(0)
    tot = np.zeros(len(seg))
    for h in harmonics:
        k = int(round(h * f0 / df))
        if k + 1 >= seg.shape[1]:
            break
        tot += 10 ** (seg[:, max(0, k - 1):k + 2].max(axis=1) / 10.0)
    return 10.0 * np.log10(tot + 1e-12)


def _no_attack(s, m, S, onset):
    """True when nothing struck pitch m at time s."""
    if CONT_TEST == "onset" and onset is not None:
        hop = INS.TRANSCRIBE_HOP_S
        fa, fb = max(0, int((s - CONT_ONSET_WIN_S) / hop)), min(len(onset), int((s + CONT_ONSET_WIN_S) / hop) + 1)
        k = m - 21
        if fb <= fa or not (0 <= k < onset.shape[1]):
            return False
        return float(onset[fa:fb, max(0, k - 1):k + 2].max()) < CONT_ONSET_MIN
    if S is None:
        return False
    S_db, df, hop_s = S
    H = _pitch_energy_db(S_db, df, m, int((s - CONT_BEFORE_S) / hop_s), int((s + CONT_AFTER_S) / hop_s) + 1)
    k_s = int(round(CONT_BEFORE_S / hop_s))
    w = int(round(CONT_DIP_S / hop_s))
    if len(H) < k_s + 4 or k_s <= w + 2:
        return False
    before, after = H[:k_s], H[k_s:]
    low = float(before[k_s - w:].min())                      # the lowest point in the last 50 ms before the start
    dip = float(np.median(before[: k_s - w])) - low          # how far it fell from the level before that
    rise = float(after.max()) - low                          # how far it rises out of it
    return rise < CONT_RISE_DB and dip < CONT_DIP_DB


def merge_continuations(notes, y, S=None, onset=None):
    """The transcriber re-triggers a held note: a pad held a bar, a power chord
    held two beats or a string line comes out as a chain of same-pitch notes
    (truth set 2026-09-09: 215 of a pad's 284 unmatched notes, 593 of a
    distorted guitar's, 235 of a string part's). A naive same-pitch merge was
    measured WRONG (repeated stabs and clav chops ARE separate notes), so the
    stem decides: a same-pitch note starting while (or within CONT_GAP_S
    after) an earlier one sounds is a continuation when nothing struck that
    pitch at its start (_no_attack: no dip-and-rise of the pitch's own
    harmonic energy) and the note it continues is long enough to be a hold
    (CONT_PRED_S). Measured on the truth set per candidate (same-pitch
    successors that are / are not truth notes): this keeps 0.97-1.00 of the
    real ones on five songs (0.84 on the funk clav) and folds 13-49 % of the
    fakes; a stricter rule (no length test, rise 12 dB) folded 80-100 % of the
    fakes and lost up to 58 % of the real notes.
    -> notes with continuations folded into the note they continue."""
    if S is None and (CONT_TEST != "onset" or onset is None or HOLD_BACKTRACK):
        S = _harmonic_frames(y)
    out = []
    last, last_len = {}, {}
    absorbed = set()
    for n in sorted(notes):
        s, e, m, a = n
        j = last.get(m)
        if j is not None and out[j][1] >= s - CONT_GAP_S and last_len[m] >= CONT_PRED_S and _no_attack(s, m, S, onset):
            ps, pe, pm, pa = out[j]
            out[j] = (ps, max(pe, e), pm, max(pa, a))
            absorbed.add(j)
            continue
        out.append((s, e, m, a))
        last[m] = len(out) - 1
        last_len[m] = e - s
    if HOLD_BACKTRACK and absorbed and S is not None:
        starts = np.array([n[0] for n in out])
        for j in sorted(absorbed):
            s, e, m, a = out[j]
            s2 = _hold_onset(s, m, S)
            if s2 < s - 0.02:
                # not before an earlier note at the same pitch
                k = np.searchsorted(starts, s) - 1
                while k >= 0 and out[k][2] != m:
                    k -= 1
                if k >= 0 and out[k][1] > s2:
                    s2 = max(s2, out[k][1])
                if s2 < s - 0.02:
                    out[j] = (s2, e, m, a)
        out.sort()
    return out


def _hold_onset(s, m, S):
    """Where the harmonic energy of pitch m rose before s: walk back while it
    stays within HOLD_BACK_DB of its level at s (at most HOLD_BACK_MAX_S), then
    place the onset at the last rise of HOLD_BACK_DB inside HOLD_BACK_RISE_S."""
    S_db, df, hop_s = S
    f_s = int(s / hop_s)
    f_a = max(0, int((s - HOLD_BACK_MAX_S) / hop_s))
    H = _pitch_energy_db(S_db, df, m, f_a, f_s + 4)
    if len(H) < 8:
        return s
    k_s = f_s - f_a
    ref = float(np.median(H[k_s: k_s + 4]))
    k = k_s
    while k > 0 and H[k - 1] >= ref - HOLD_BACK_DB:
        k -= 1
    if k >= k_s - 2:
        return s
    # the rise: the earliest frame from k on where the energy climbs HOLD_BACK_DB within HOLD_BACK_RISE_S
    w = max(1, int(HOLD_BACK_RISE_S / hop_s))
    for q in range(max(0, k - w), k_s):
        if H[min(len(H) - 1, q + w)] - H[q] >= HOLD_BACK_DB:
            return (f_a + q) * hop_s
    return (f_a + k) * hop_s


def drop_ghosts(notes, y, S=None):
    """A note an octave under a real one, transcribed from the real note's
    even harmonics: over the note, the energy at its own odd harmonics (1, 3,
    5 - which the real note does not have) sits GHOST_ODD_DB or more under the
    even ones (2, 4, 6 - the real note's 1, 2, 3). Only notes that sound while
    a note an octave above does are tested."""
    if S is None:
        S = _harmonic_frames(y)
    S_db, df, hop_s = S
    notes = sorted(notes)
    starts = np.array([n[0] for n in notes])
    out = []
    for i, (s, e, m, a) in enumerate(notes):
        lo, hi = np.searchsorted(starts, s - 2.0), np.searchsorted(starts, e)
        above = any(notes[j][2] == m + 12 and notes[j][1] > s + 0.02 and notes[j][0] < e - 0.02 for j in range(lo, hi) if j != i)
        if above:
            fa, fb = int((s + 0.01) / hop_s), int(min(e, s + 0.5) / hop_s) + 1
            odd = _pitch_energy_db(S_db, df, m, fa, fb, harmonics=(1, 3, 5))
            even = _pitch_energy_db(S_db, df, m, fa, fb, harmonics=(2, 4, 6))
            if len(odd) and float(np.median(odd - even)) <= GHOST_ODD_DB:
                continue
        out.append((s, e, m, a))
    return out


def _merge_fragments(notes):
    """The transcriber re-triggers a held note: the same pitch again after a
    short gap (a pad held a bar comes out as several notes, only the first
    at the true onset - pop truth track: precision 0.41). Same pitch, gap
    under FRAG_GAP_S -> one note."""
    by = {}
    for n in notes:
        by.setdefault(n[2], []).append(n)
    out = []
    for m, lst in by.items():
        lst.sort()
        cur = list(lst[0])
        for s, e, _m, a in lst[1:]:
            if s - cur[1] <= FRAG_GAP_S:
                cur[1] = max(cur[1], e)
                cur[3] = max(cur[3], a)
            else:
                out.append(tuple(cur))
                cur = [s, e, m, a]
        out.append(tuple(cur))
    return sorted(out)


def _groups(notes):
    """notes [(t0, t1, midi, amp)] sorted -> [[notes...]] grouped by start."""
    groups = []
    for n in sorted(notes):
        if groups and n[0] - groups[-1][0][0] <= CHORD_S:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _harmonic_profile(y, t0, midis, post_s=0.13, pre_s=0.2):
    """The group's harmonic profile: for each of its notes, the spectrum after
    the onset minus the spectrum before it (what was already sounding), read
    at the harmonics of that note; averaged over the notes -> 12 dB values
    relative to the loudest harmonic. Pitch-invariant, so one instrument
    stays one whatever it plays (the old reader's split rule: 6 dB rms
    between profiles)."""
    s = int(t0 * RATE)
    a0, a1 = max(0, s - int(pre_s * RATE)), max(0, s - int(0.005 * RATE))
    b0, b1 = s + int(0.005 * RATE), min(len(y), s + int(post_s * RATE))
    if b1 - b0 < 1024:
        return np.full(12, -30.0)
    Post, df = INS._spectrum(y[b0:b1], INS._N_HPS)
    if a1 - a0 >= 1024:
        Pre, _df = INS._spectrum(y[a0:a1], INS._N_HPS)
        R = np.maximum(Post - 1.2 * Pre, 0.0)
    else:
        R = Post
    profs = []
    for m in midis:
        f0 = 440.0 * 2 ** ((m - 69) / 12.0)
        vals = []
        for h in range(1, 13):
            i = int(round(h * f0 / df))
            vals.append(float(R[i]) if i < len(R) else 0.0)
        v = 20 * np.log10(np.array(vals) + 1e-9)
        profs.append(np.clip(v - v.max(), -60.0, 0.0))
    return np.mean(profs, axis=0)


def _features(y, groups):
    """Per group: the harmonic profile (12 dB), the sustained-vs-attack mel change, duration, register, size."""
    import librosa
    feats, levels = [], []
    for g in groups:
        t0 = g[0][0]
        a, b = int(t0 * RATE), int(min(len(y), (t0 + ATTACK_S) * RATE))
        seg = y[a:b]
        if len(seg) < 1024:
            seg = np.concatenate([seg, np.zeros(1024 - len(seg), dtype=np.float32)])
        prof = _harmonic_profile(y, t0, [n[2] for n in g])
        M = librosa.feature.melspectrogram(y=np.ascontiguousarray(seg, dtype=np.float32), sr=RATE, n_fft=1024, hop_length=256, n_mels=40,
                                           fmin=30.0, fmax=12000.0, power=1.0).mean(axis=1)
        L = 20 * np.log10(M + 1e-6)
        L -= L.max()
        # the sustained part relative to the attack: how the sound holds (a pad vs a stab) - per band
        a2, b2 = int((t0 + SUSTAIN_FROM_S) * RATE), int(min(len(y), (t0 + SUSTAIN_TO_S) * RATE))
        seg2 = y[a2:b2] if b2 - a2 >= 1024 else np.zeros(1024, dtype=np.float32)
        M2 = librosa.feature.melspectrogram(y=np.ascontiguousarray(seg2, dtype=np.float32), sr=RATE, n_fft=1024, hop_length=256, n_mels=40,
                                            fmin=30.0, fmax=12000.0, power=1.0).mean(axis=1)
        hold = np.clip(20 * np.log10(M2 + 1e-6) - 20 * np.log10(M + 1e-6), -40.0, 10.0)
        dur = float(np.median([n[1] - n[0] for n in g]))
        mean_midi = float(np.mean([n[2] for n in g]))
        if FEATURES == "profile":
            # dB features left in their own units (PROFILE_SEP is a dB distance): the harmonic profile,
            # a coarse hold-vs-attack change (8 bands), and length / register / size on dB-like scales
            hold8 = hold.reshape(8, 5).mean(axis=1)
            feats.append(np.concatenate([prof, hold8 * 0.5, [np.log1p(dur) * 12.0, mean_midi / 2.0, min(len(g), 4) * 4.0]]))
        else:
            feats.append(np.concatenate([L / 10.0, hold / 10.0, [np.log1p(dur) * 3.0, mean_midi / 6.0, min(len(g), 4) * 1.5]]))
        levels.append(float(np.sqrt(np.mean(seg.astype(np.float64) ** 2))))
    return np.array(feats), np.array(levels)


def _merge_by_cooccurrence(groups, labels, F):
    """Two clusters whose notes never sound over each other and whose
    profiles sit within MERGE_PROFILE_DB are one instrument split by
    register or velocity; two that often sound together are two."""
    labs = sorted(set(labels.tolist()))
    if len(labs) < 2:
        return labels
    spans = {l: [(g[0][0], max(n[1] for n in g)) for i, g in enumerate(groups) if labels[i] == l] for l in labs}
    cent = {l: F[labels == l].mean(axis=0) for l in labs}
    changed = True
    while changed and len(labs) > 1:
        changed = False
        best = None
        for a in labs:
            for b in labs:
                if b <= a:
                    continue
                d = float(np.sqrt(np.mean((cent[a][:12] - cent[b][:12]) ** 2)))
                if d > MERGE_PROFILE_DB:
                    continue
                sb = np.array(sorted(spans[b]))
                over = 0
                for t0, t1 in spans[a]:
                    k = np.searchsorted(sb[:, 0], t0)
                    if (k < len(sb) and sb[k, 0] < t1 - 0.02) or (k > 0 and sb[k - 1, 1] > t0 + 0.02):
                        over += 1
                rate = over / max(len(spans[a]), 1)
                if rate <= MERGE_OVERLAP and (best is None or d < best[0]):
                    best = (d, a, b)
        if best is not None:
            _d, a, b = best
            labels = np.where(labels == b, a, labels)
            spans[a] = spans[a] + spans.pop(b)
            cent[a] = F[labels == a].mean(axis=0)
            labs.remove(b)
            changed = True
    return labels


PROGRAM_MIN_NOTES = 12         # a GM program with fewer notes than this folds into the nearest program family
SPLIT_CHORD_LEVEL = True       # a group's measured level is the CHORD's; each of its tones gets an equal share (1/n of the
                               # energy), or a three-note chord renders as three chords (truth set 2026-09-09: the ballad's
                               # additive piano 9 dB above the true stem, sample voices 4-8 dB hot on polyphonic parts)


def _program_family(program):
    """GM program -> a coarse family name (the voice's hint)."""
    p = int(program)
    if p < 8:
        return "piano"
    if p < 16:
        return "mallet"
    if p < 24:
        return "organ"
    if p < 32:
        return "guitar"
    if p < 40:
        return "bass"
    if p < 48:
        return "strings"
    if p < 56:
        return "ensemble"
    if p < 64:
        return "brass"
    if p < 72:
        return "reed"
    if p < 80:
        return "pipe"
    if p < 88:
        return "lead"
    if p < 96:
        return "pad"
    if p < 104:
        return "fx"
    return "perc"


def read_labelled(y, beats, period, labelled, stem="other", progress=None):
    """Notes that come with an instrument label (YourMT3+: [(on, off, midi, program, is_drum, vel)])
    -> instruments, one voice per GM program (small programs fold into the nearest), the events
    built exactly as read() builds them from basic-pitch notes."""
    y = INS._mono(y)
    lo, hi = INS.PITCH_RANGE.get(stem, (24, 108))
    notes = [(float(s), float(e), int(m), 1.0, int(p)) for s, e, m, p, is_drum, _v in labelled
             if not is_drum and lo <= m <= hi and (e - s) >= MIN_NOTE_S]
    if len(notes) < 8:
        return []
    by_prog = {}
    for n in notes:
        by_prog.setdefault(n[4], []).append(n)
    # small programs fold into the nearest program number of a large one
    big = sorted(p for p, lst in by_prog.items() if len(lst) >= PROGRAM_MIN_NOTES) or [max(by_prog, key=lambda p: len(by_prog[p]))]
    voice_notes = {p: [] for p in big}
    for p, lst in by_prog.items():
        target = p if p in voice_notes else min(big, key=lambda q: abs(q - p))
        voice_notes[target].extend(lst)
    if progress:
        progress(f"{stem}: {len(notes)} labelled notes -> {len(voice_notes)} program voices " + ", ".join(f"{_program_family(p)}({p}) {len(v)}" for p, v in voice_notes.items()))
    S = _harmonic_frames(y) if (CONT_MERGE or GHOST_DROP) else None
    beats = np.asarray(beats, dtype=np.float64)
    out = []
    for prog, lst in sorted(voice_notes.items(), key=lambda kv: -len(kv[1])):
        vnotes = [(s, e, m, a) for s, e, m, a, _p in lst]
        if CONT_MERGE:
            vnotes = merge_continuations(vnotes, y, S)
        if GHOST_DROP:
            vnotes = drop_ghosts(vnotes, y, S)
        groups = _groups(vnotes)
        if len(groups) < 4:
            continue
        _F, levels = _features(y, groups)
        inst = _voice_from_groups(y, groups, levels, list(range(len(groups))), beats, period, stem)
        if inst is None:
            continue
        inst["hint"] = _program_family(prog)
        inst["program"] = int(prog)
        inst["reader"] = "ymt3"
        out.append(inst)
    if out:
        levels_db = INS._level_rel([i["level_dbfs"] for i in out])
        for inst, l in zip(out, levels_db):
            inst["level_db"] = l
    out.sort(key=lambda d: -d["n"])
    return out


def _voice_from_groups(y, groups, levels, idx, beats, period, stem):
    """One instrument record from a set of groups (shared by read() and read_labelled())."""
    lv = levels[idx]
    ref = float(np.percentile(lv[lv > 0], 95)) if np.any(lv > 0) else 1.0
    events, durs = [], []
    for i in idx:
        g = groups[i]
        t0 = g[0][0]
        pos = INS.beat_step_of(beats, t0, period)
        if pos is None:
            continue
        b, st = pos
        grid_t = INS.event_time(beats, b, st)
        lvl = max(levels[i], 1e-9) / (np.sqrt(len(g)) if (SPLIT_CHORD_LEVEL and len(g) > 1) else 1.0)
        vel = float(np.clip(1.0 + 20 * np.log10(lvl / ref) / VEL_RANGE_DB, 0.15, 1.0))
        for s, e, m, a in g:
            d = max(1, int(round((e - s) / (period / STEPS))))
            durs.append(e - s)
            events.append([int(b), int(st), int(m), int(d), round(vel, 3), round(float(min(a, 1.0)), 3), round(float(t0 - grid_t), 4)])
    if len(events) < 4:
        return None
    events.sort(key=lambda e: (e[0], e[1], e[2]))
    held = float(np.mean([d >= HELD_S for d in durs]))
    kind = "sustain" if held >= 0.5 else "hit"
    best = None
    for i in idx:
        t0 = groups[i][0][0]
        nxt = groups[i + 1][0][0] if i + 1 < len(groups) else t0 + 10.0
        if nxt - t0 >= 0.2 and (best is None or levels[i] > best[0]):
            best = (levels[i], t0, min(nxt, t0 + 1.0), min(n[2] for n in groups[i]))
    if best is None:
        i = max(idx, key=lambda j: levels[j])
        t0 = groups[i][0][0]
        best = (levels[i], t0, t0 + 0.5, min(n[2] for n in groups[i]))
    import librosa
    ex = y[int(best[1] * RATE):int(best[2] * RATE)]
    cen = float(librosa.feature.spectral_centroid(y=np.ascontiguousarray(ex), sr=RATE).mean()) if len(ex) > 2048 else 500.0
    ms = [e[2] for e in events]
    poly = float(np.mean([len(groups[i]) > 1 for i in idx]))
    hint = ("pad" if poly >= 0.5 else "lead") if kind == "sustain" else ("stab" if poly >= 0.5 else "pluck")
    inst = {"stem": stem, "kind": kind, "pitched": True, "range": [int(min(ms)), int(max(ms))], "centroid_hz": round(cen, 1),
            "decay_db": 0.0, "dur_s": round(float(np.median(durs)), 3), "n": len(events), "level_db": 0.0,
            "level_dbfs": round(float(20 * np.log10(ref + 1e-9)), 1),
            "exemplar": [round(float(best[1]), 4), round(float(best[2]), 4)], "exemplar_midi": int(best[3]),
            "hint": hint, "reader": "transcription", "events": events}
    if kind == "sustain":
        inst["hold_decay_db_s"] = 0.0
    return inst


def read(y, beats, period, stem="other", progress=None, max_voices=None):
    y = INS._mono(y)
    k_max = max_voices or MAX_VOICES.get(stem, 4)
    tr, onset = INS.transcribe_full(y)
    if not tr:
        return []
    notes = [(float(s), float(e), int(m), float(a)) for s, e, m, a in tr if a >= AMP_MIN and (e - s) >= MIN_NOTE_S]
    lo, hi = INS.PITCH_RANGE.get(stem, (24, 108))
    notes = [n for n in notes if lo <= n[2] <= hi]
    if FRAG_GAP_S > 0:
        notes = _merge_fragments(notes)
    S = None
    if CONT_MERGE or GHOST_DROP:
        S = _harmonic_frames(y) if (GHOST_DROP or CONT_TEST != "onset" or onset is None) else None
        n0 = len(notes)
        if CONT_MERGE and CONT_SCOPE == "all":
            notes = merge_continuations(notes, y, S, onset=onset)
        n1 = len(notes)
        if GHOST_DROP:
            notes = drop_ghosts(notes, y, S)
        if progress:
            progress(f"{stem}: {n0 - n1} continuations folded, {n1 - len(notes)} sub-octave ghosts dropped")
    if len(notes) < 8:
        return []
    groups = _groups(notes)
    F, levels = _features(y, groups)
    if len(groups) < 16:
        labels = np.zeros(len(groups), dtype=int)
    elif FEATURES == "profile":
        labels = INS._cluster_auto(F, k_max, min_share=0.05, min_sil=CLUSTER_MIN_SIL, standardize=False, min_sep=INS.PROFILE_SEP)
        labels = _merge_by_cooccurrence(groups, labels, F)
    else:
        labels = _cluster_groups(F, k_max, stem)
    if progress:
        progress(f"{stem}: transcription {len(notes)} notes in {len(groups)} groups -> {len(set(labels.tolist()))} voices")
    beats = np.asarray(beats, dtype=np.float64)
    out = []
    n_folded = 0
    for lab in sorted(set(labels.tolist())):
        idx = [i for i in range(len(groups)) if labels[i] == lab]
        level_at = {groups[i][0][0]: float(levels[i]) for i in idx}
        vgroups = [groups[i] for i in idx]
        if CONT_MERGE and CONT_SCOPE == "voice":
            # continuations folded WITHIN the voice: a same-pitch predecessor from another voice (an e-piano
            # chord under a clav chop on the same pitches) can no longer absorb a real note
            vnotes = [n for g in vgroups for n in g]
            merged = merge_continuations(vnotes, y, S, onset=onset)
            n_folded += len(vnotes) - len(merged)
            vgroups = _groups(merged)
        vlevels = np.array([level_at.get(g[0][0], 0.0) for g in vgroups])
        lv = vlevels
        ref = float(np.percentile(lv[lv > 0], 95)) if np.any(lv > 0) else 1.0
        events = []
        durs = []
        for gi, g in enumerate(vgroups):
            t0 = g[0][0]
            pos = INS.beat_step_of(beats, t0, period)
            if pos is None:
                continue
            b, st = pos
            grid_t = INS.event_time(beats, b, st)
            lvl = max(vlevels[gi], 1e-9) / (np.sqrt(len(g)) if (SPLIT_CHORD_LEVEL and len(g) > 1) else 1.0)
            vel = float(np.clip(1.0 + 20 * np.log10(lvl / ref) / VEL_RANGE_DB, 0.15, 1.0))
            for s, e, m, a in g:
                d = max(1, int(round((e - s) / (period / STEPS))))
                durs.append(e - s)
                events.append([int(b), int(st), int(m), int(d), round(vel, 3), round(float(min(a, 1.0)), 3), round(float(t0 - grid_t), 4)])
        if len(events) < 4:
            continue
        events.sort(key=lambda e: (e[0], e[1], e[2]))
        held = float(np.mean([d >= HELD_S for d in durs]))
        kind = "sustain" if held >= 0.5 else "hit"
        # exemplar: the loudest group with room after it
        best = None
        for i in idx:
            t0 = groups[i][0][0]
            nxt = groups[i + 1][0][0] if i + 1 < len(groups) else t0 + 10.0
            if nxt - t0 >= 0.2 and (best is None or levels[i] > best[0]):
                best = (levels[i], t0, min(nxt, t0 + 1.0), min(n[2] for n in groups[i]))
        if best is None:
            i = max(idx, key=lambda j: levels[j])
            t0 = groups[i][0][0]
            best = (levels[i], t0, t0 + 0.5, min(n[2] for n in groups[i]))
        import librosa
        ex = y[int(best[1] * RATE):int(best[2] * RATE)]
        cen = float(librosa.feature.spectral_centroid(y=np.ascontiguousarray(ex), sr=RATE).mean()) if len(ex) > 2048 else 500.0
        ms = [e[2] for e in events]
        poly = float(np.mean([len(groups[i]) > 1 for i in idx]))
        hint = ("pad" if poly >= 0.5 else "lead") if kind == "sustain" else ("stab" if poly >= 0.5 else "pluck")
        inst = {"stem": stem, "kind": kind, "pitched": True, "range": [int(min(ms)), int(max(ms))], "centroid_hz": round(cen, 1),
                "decay_db": 0.0, "dur_s": round(float(np.median(durs)), 3), "n": len(events), "level_db": 0.0,
                "level_dbfs": round(float(20 * np.log10(ref + 1e-9)), 1),
                "exemplar": [round(float(best[1]), 4), round(float(best[2]), 4)], "exemplar_midi": int(best[3]),
                "hint": hint, "reader": "transcription", "events": events}
        if kind == "sustain":
            inst["hold_decay_db_s"] = 0.0
        out.append(inst)
    if progress and n_folded:
        progress(f"{stem}: {n_folded} continuations folded within their voices")
    levels_db = INS._level_rel([i["level_dbfs"] for i in out])
    for inst, l in zip(out, levels_db):
        inst["level_db"] = l
    out.sort(key=lambda d: -d["n"])
    return out
