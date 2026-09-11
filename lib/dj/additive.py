"""Additive (harmonic) instrument models learned from the song's own notes.

A pitched voice is described by how the amplitudes of its harmonics move
in time: an ATTACK matrix (each harmonic's level over the first 0.25 s,
relative), a DECAY rate per harmonic (dB/s while the note holds) and a
RELEASE time - measured on every note the voice plays in the song and
averaged (median), so the profile is the song's instrument, not a
preset. A note at any pitch and length is rendered by adding sinusoids
at the harmonics of its fundamental with those envelopes, at the level
the reading gives the event.

This is the representation that stays MORPHABLE: change the profile and
the same notes play on a different tone; change the notes and the
profile plays them. A single recorded sample can do neither.

What it will not capture: noise components (breath, filter hiss),
inharmonic partials, detuned oscillator beating; those are the
residual, and the evaluator says how much of a voice they were.
"""
import numpy as np

from lib.dj import instruments as INS

RATE = INS.RATE
STEPS = INS.STEPS
N_HARM = 24
HOP = 256
N_FFT = 2048
ATTACK_S = 0.25
ATTACK_FRAMES = int(ATTACK_S * RATE / HOP)
MIN_NOTES = 4
MASK_DB = 6.0                  # another voice within this much of the stem at one of our harmonics: that reading is masked
SUSTAIN_PCT = 20               # a held voice's harmonic level = this percentile over its sustained frames (transients above it are others')


def _hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12.0)


def _harmonic_track(S, freqs, f0, n_harm=N_HARM):
    """S: |STFT| (bins, frames). -> (n_harm, frames) magnitude at each
    harmonic (peak within +-1 bin of h*f0, quadratic-free)."""
    out = np.zeros((n_harm, S.shape[1]), dtype=np.float32)
    df = freqs[1] - freqs[0]
    for h in range(1, n_harm + 1):
        f = h * f0
        if f >= freqs[-1] - 2 * df:
            break
        i = int(round(f / df))
        lo, hi = max(0, i - 1), min(len(freqs), i + 2)
        out[h - 1] = S[lo:hi].max(axis=0)
    return out


def analyze_voice(inst, stem_y, result, max_notes=120, progress=None, others_y=None):
    """-> profile {"attack": (N_HARM, ATTACK_FRAMES) dB rel. the note's peak harmonic,
    "decay_db_s": (N_HARM,) dB/s during the hold, "release_s", "shape": (N_HARM,) dB
    the sustained spectrum, "n_notes"} or None when the voice has too few clean notes.
    Notes of the same voice sounding at the same time are skipped as
    sources (their harmonics collide).
    others_y: the OTHER voices of the stem rendered from the program (same
    length as stem_y). Where their magnitude at one of this voice's
    harmonics is within MASK_DB of the stem's, that harmonic is not read
    in that frame (a melody an octave above a pad sits on the pad's second
    harmonic and made the profile 7 dB too loud and far too bright on the
    synthetic gate). A spectral MASK, not a waveform subtraction: a sample
    render is not phase-aligned to the recording, so subtracting it removes
    little and can add."""
    import librosa
    y = np.asarray(stem_y, dtype=np.float32)
    oy = None if others_y is None else np.asarray(others_y, dtype=np.float32)
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or 0.5)
    events = [e for e in inst["events"] if e[2] is not None]
    if len(events) < MIN_NOTES:
        return None
    # the best-isolated notes: loudest first, no other note of this voice within its span
    times = [(INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0), e) for e in events]
    times.sort()
    starts = np.array([t for t, _e in times])
    cands = []
    for k, (t, e) in enumerate(times):
        length = e[3] * period / STEPS
        nxt = starts[k + 1] - t if k + 1 < len(starts) else 3.0
        if nxt < 0.12:
            continue
        cands.append((e[4] + 0.5 * min(nxt, 1.0), t, e, min(nxt, length + 0.3, 3.0)))
    cands.sort(reverse=True)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / RATE)
    attacks, decays, shapes = [], [], []
    ref_levels = []
    for _s, t, e, span in cands[:max_notes]:
        a = max(0, int((t - 0.005) * RATE))
        b = min(len(y), int((t + span) * RATE))
        seg = y[a:b]
        if len(seg) < N_FFT:
            continue
        S = np.abs(librosa.stft(seg, n_fft=N_FFT, hop_length=HOP))
        H = _harmonic_track(S, freqs, _hz(int(e[2])))
        Hdb = 20 * np.log10(H + 1e-7)
        if oy is not None:
            So = np.abs(librosa.stft(oy[a:b], n_fft=N_FFT, hop_length=HOP))
            Ho = _harmonic_track(So, freqs, _hz(int(e[2])))
            Hdb[Ho >= H * 10 ** (-MASK_DB / 20.0)] = np.nan        # another voice owns this harmonic here
        if np.all(np.isnan(Hdb[:, : ATTACK_FRAMES + 2])):
            continue
        # the level reference: the FUNDAMENTAL over the note's sustained frames (median), not the
        # loudest harmonic in the attack - another voice an octave up sits on the second harmonic and
        # made the reference (and the whole render) 7 dB too loud on the synthetic gate; a unison
        # collision on the fundamental is rarer. Harmonics may exceed the reference (up to +12 dB).
        sus = Hdb[0, ATTACK_FRAMES // 2: max(ATTACK_FRAMES, Hdb.shape[1])]
        sus = sus[np.isfinite(sus)]
        if not len(sus):
            continue
        pk = float(np.median(sus))
        if pk < -70:
            continue
        rel = Hdb - pk
        if inst.get("kind") == "sustain" and Hdb.shape[1] > ATTACK_FRAMES + 4:
            # a HELD voice's harmonic levels: a low percentile over the sustained frames, per harmonic.
            # Other voices' notes on the same harmonics are transient (a pluck decays between its
            # hits); the pad's own level is the floor they decay back to. The median read the melody
            # an octave up as the pad's second harmonic at +17 dB (synthetic gate: pads +7 dB loud).
            # The attack keeps the fundamental's measured rise, applied to every harmonic.
            with np.errstate(all="ignore"):
                steady = np.nanpercentile(rel[:, ATTACK_FRAMES:], SUSTAIN_PCT, axis=1)
            steady = np.where(np.isnan(steady), -90.0, steady)
            rise = rel[0, :ATTACK_FRAMES]
            rise = np.where(np.isfinite(rise), rise, 0.0) - float(steady[0])
            rel = np.concatenate([steady[:, None] + np.minimum(rise, 0.0)[None, :], np.repeat(steady[:, None], rel.shape[1] - ATTACK_FRAMES, axis=1)], axis=1)
        att = rel[:, :ATTACK_FRAMES]
        if att.shape[1] < ATTACK_FRAMES:
            att = np.concatenate([att, np.repeat(att[:, -1:], ATTACK_FRAMES - att.shape[1], axis=1)], axis=1)
        attacks.append(att)
        # decay per harmonic: slope over the hold after the attack (only where the harmonic is audible)
        n = rel.shape[1]
        if n > ATTACK_FRAMES + 6:
            xs = np.arange(ATTACK_FRAMES, n) * HOP / RATE
            d = np.full(N_HARM, np.nan)
            for h in range(N_HARM):
                ys = rel[h, ATTACK_FRAMES:n]
                ok = np.isfinite(ys) & (ys > -60)
                if ok.sum() >= 6:
                    d[h] = np.polyfit(xs[ok], ys[ok], 1)[0]
            decays.append(d)
        with np.errstate(all="ignore"):
            shapes.append(np.nanmean(rel[:, ATTACK_FRAMES // 2: ATTACK_FRAMES], axis=1))
        ref_levels.append((pk, e[4]))
    if len(attacks) < MIN_NOTES:
        return None
    with np.errstate(all="ignore"):
        attack = np.nanmedian(np.array(attacks), axis=0)
        decay = np.nanmedian(np.array(decays), axis=0) if decays else np.full(N_HARM, -30.0)
    # a harmonic never read clean anywhere is not part of the voice
    attack = np.where(np.isnan(attack), -90.0, attack)
    decay = np.where(np.isnan(decay), -30.0, np.clip(decay, -400.0, -1.0))
    if inst.get("kind") == "sustain" and inst.get("hold_decay_db_s") is not None:
        # a held voice's level over its hold is in the reading (per-beat levels of every hold): use
        # that slope for every harmonic. The harmonic fit on the stem picks up the other voices'
        # decaying notes on the same harmonics (a flat pad measured -6 dB/s on the synthetic gate).
        decay = np.full(N_HARM, float(np.clip(inst["hold_decay_db_s"], -60.0, 20.0)), dtype=np.float32)
    with np.errstate(all="ignore"):
        shape = np.nanmedian(np.array(shapes), axis=0)
    shape = np.where(np.isnan(shape), -90.0, shape)
    # the level reference: peak harmonic dB at velocity 1 (undo each note's velocity, median)
    lv = np.median([pk - (v - 1.0) * 24.0 for pk, v in ref_levels])
    prof = {"attack": attack.astype(np.float32), "decay_db_s": decay.astype(np.float32), "shape": shape.astype(np.float32),
            "release_s": 0.06, "peak_db": float(lv), "n_notes": len(attacks)}
    if progress:
        progress(f"{inst['id']}: additive profile from {len(attacks)} notes, decay h1 {decay[0]:.0f} dB/s")
    return prof


def render_note(prof, midi, dur_s, vel=1.0, phase_seed=0, min_decay_db_s=None):
    """Additive synthesis of one note: harmonics of the fundamental with the
    profile's attack envelopes, then per-harmonic exponential decay until
    dur_s, then the release. -> mono float32 at the profile's level x vel.
    min_decay_db_s: the decay clamped to at least this (e.g. -6: a HELD
    articulation of a struck voice's profile sustains instead of dying).

    The envelopes are built in dB on the analysis hop grid for all harmonics
    at once and interpolated linearly in amplitude to the sample rate; the
    sinusoids come from the angle-addition recurrence on the fundamental
    (sin/cos of h*theta from those of (h-1)*theta), so one sin and one cos
    over the note replace one sin per harmonic. Same sound as the reference
    routine below to a fraction of a dB (the release ramps between hops
    instead of per sample); 5-10x faster, and this call was 90% of a
    program build and of every additive render."""
    f0 = _hz(int(midi))
    decay_db_s = np.asarray(prof["decay_db_s"], dtype=np.float64)
    if min_decay_db_s is not None:
        decay_db_s = np.maximum(decay_db_s, min_decay_db_s)
    n_att = ATTACK_FRAMES
    hold_s = max(0.0, dur_s - ATTACK_S)
    rel_s = float(prof["release_s"])
    total = int((ATTACK_S + hold_s + rel_s) * RATE) + 1
    rng = np.random.default_rng(phase_seed)
    attack = np.asarray(prof["attack"], dtype=np.float64)
    hs = []
    for h in range(1, N_HARM + 1):
        if h * f0 >= RATE / 2 * 0.95:
            break
        if attack[h - 1].max() < -70:
            hs.append((h, None))                       # silent: stepped over, no phase drawn (as before)
            continue
        hs.append((h, rng.uniform(0, 2 * np.pi)))
    play = [(h, ph) for h, ph in hs if ph is not None]
    if not play:
        return np.zeros(total, dtype=np.float32)
    # --- envelopes in dB on the hop grid: (H, n_c), n_c hops cover the note plus one ---
    n_c = total // HOP + 2
    tc = np.arange(n_c) * (HOP / RATE)
    idx = np.array([h - 1 for h, _ph in play])
    env_db = np.empty((len(play), n_c), dtype=np.float64)
    n_a = min(n_att, n_c)
    env_db[:, :n_a] = attack[idx, :n_a]
    if n_c > n_att:
        env_db[:, n_att:] = attack[idx, -1:] + decay_db_s[idx, None] * (tc[n_att:] - tc[n_att - 1])[None, :]
    np.minimum(env_db, 12.0, out=env_db)               # a swelling hold may not run away
    rel_start = ATTACK_S + hold_s
    rel = tc > rel_start
    env_db[:, rel] -= (tc[rel] - rel_start)[None, :] / max(rel_s, 0.01) * 60.0
    amp = (10.0 ** (np.clip(env_db, -90.0, 12.0) / 20.0)).astype(np.float32)
    # --- linear interpolation in amplitude to the sample rate: (H, (n_c-1)*HOP) -> [:total] ---
    w = (np.arange(HOP, dtype=np.float32) / HOP)[None, None, :]
    env = (amp[:, :-1, None] * (1.0 - w) + amp[:, 1:, None] * w).reshape(len(play), -1)[:, :total]
    # --- sinusoids by recurrence on the fundamental ---
    theta = (2 * np.pi * f0 / RATE) * np.arange(total, dtype=np.float64)
    s1, c1 = np.sin(theta).astype(np.float32), np.cos(theta).astype(np.float32)
    out = np.zeros(total, dtype=np.float32)
    s_h, c_h = s1.copy(), c1.copy()                    # h = 1
    ph_of = dict(hs)
    k = 0
    tmp = np.empty(total, dtype=np.float32)
    for h, _ph in hs:
        if h > 1:
            # (s, c)_h = (s c1 + c s1, c c1 - s s1)
            np.multiply(s_h, c1, out=tmp)
            tmp += c_h * s1
            c_h = c_h * c1 - s_h * s1
            s_h, tmp = tmp, s_h
        ph = ph_of[h]
        if ph is None:
            continue
        # sin(h*theta + ph) = s_h cos(ph) + c_h sin(ph)
        out += env[k] * (s_h * np.float32(np.cos(ph)) + c_h * np.float32(np.sin(ph)))
        k += 1
    scale = 10 ** ((prof["peak_db"] + (vel - 1.0) * 24.0) / 20.0) * (2.0 / N_FFT) * 2.0
    return out * np.float32(scale)


def render_note_ref(prof, midi, dur_s, vel=1.0, phase_seed=0, min_decay_db_s=None):
    """The per-sample, per-harmonic reference synthesis render_note replaces (kept for the A/B)."""
    f0 = _hz(int(midi))
    decay_db_s = prof["decay_db_s"] if min_decay_db_s is None else np.maximum(prof["decay_db_s"], min_decay_db_s)
    n_att = ATTACK_FRAMES
    hold_s = max(0.0, dur_s - ATTACK_S)
    total = int((ATTACK_S + hold_s + prof["release_s"]) * RATE) + 1
    t = np.arange(total) / RATE
    out = np.zeros(total, dtype=np.float32)
    rng = np.random.default_rng(phase_seed)
    frame_t = np.arange(n_att) * HOP / RATE
    for h in range(1, N_HARM + 1):
        f = h * f0
        if f >= RATE / 2 * 0.95:
            break
        att = prof["attack"][h - 1]
        if att.max() < -70:
            continue
        env_db = np.interp(t, frame_t, att, right=att[-1])
        after = t > frame_t[-1]
        env_db[after] = att[-1] + decay_db_s[h - 1] * (t[after] - frame_t[-1])
        env_db = np.minimum(env_db, 12.0)                # a swelling hold may not run away
        # release
        rel_start = ATTACK_S + hold_s
        rel = t > rel_start
        env_db[rel] -= (t[rel] - rel_start) / max(prof["release_s"], 0.01) * 60.0
        env = 10 ** (np.clip(env_db, -90.0, 12.0) / 20.0)
        out += (env * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))).astype(np.float32)
    # level: the profile's peak harmonic is at peak_db (magnitude in an N_FFT window): convert the
    # sinusoid amplitude so a full-scale harmonic matches the analysis scale
    scale = 10 ** ((prof["peak_db"] + (vel - 1.0) * 24.0) / 20.0) * (2.0 / N_FFT) * 2.0
    return out * np.float32(scale)


def residual_level(inst, stem_y, result, prof, n=12):
    """How much of the voice the harmonic model leaves: for n loud notes,
    the energy of (stem - additive render) over the note relative to the
    stem's energy there (dB; 0 = nothing explained, -10 = 90% explained)."""
    y = np.asarray(stem_y, dtype=np.float32)
    beats = np.asarray(result["beats"], dtype=np.float64)
    period = float(result.get("period_s") or 0.5)
    events = sorted([e for e in inst["events"] if e[2] is not None], key=lambda e: -e[4])[:n]
    ratios = []
    for e in events:
        t = INS.event_time(beats, e[0], e[1]) + (e[6] if len(e) > 6 else 0.0)
        dur = min(1.0, e[3] * period / STEPS)
        a, b = int(t * RATE), int(min(len(y), (t + dur) * RATE))
        if b - a < 2048:
            continue
        r = render_note(prof, e[2], dur, e[4])[: b - a]
        seg = y[a:a + len(r)]
        # align phase-free: compare magnitude spectra
        Sa = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        Sb = np.abs(np.fft.rfft(r * np.hanning(len(r))))
        ratios.append(float(np.sum(np.maximum(Sa - Sb, 0) ** 2) / (np.sum(Sa ** 2) + 1e-12)))
    return 10 * np.log10(np.median(ratios) + 1e-9) if ratios else 0.0
