"""Verify a rendered seam with EXACT beat bookkeeping - no detection.

Built to the user's spec (2026-08-04), in their own terms:

  "preknowledge of the songs via direct analysis of every beat"
      -> beat TIMES come from each track's stored grid, never from onset
         detection (three detector-based instruments failed the same day:
         they measure hats, fills and vocals, not beats).

  "the overlap is only going to make sense if you've already done beat
   matching speed/tone transforms"
      -> beats are projected into RENDER time through each deck's actual
         position trace - recorded through the real stretch, rate ramps
         and PLL trims - so alignment is judged on the transformed audio,
         exactly as heard.

  "we know where the power would be"
      -> each track must CONCENTRATE its low-band power on its own beats
         in its solo span. A wrong grid fails this (power off-beat), and
         so does diffuse material (a percussion wall with power
         everywhere) - the pair whose signatures claimed kick agreement
         0.99 while rendering as a 3.2x low-end mess fails HERE, because
         beat-matching material with no localized beat power is
         meaningless however good the metadata looks.

Two instruments, both pure arithmetic on known quantities:

  GRID HONESTY (per track, solo span): mean low-band envelope at the
      track's own beat times vs everywhere. Below ~1.3x, the track has no
      beat to match - either the grid is wrong or the material is
      diffuse. Refuse to judge (and refuse to blend) rather than guess.

  DECK ALIGNMENT (dual span): every A-beat's distance to the nearest
      B-beat, in render time. Median within flam tolerance = matched.
      A wrong tempo MULTIPLE shows up here as a sawtooth of deltas.

  PERCUSSION CO-PRESENCE (perc_overlap, 2026-08-06): how long BOTH decks
      are percussively live at once. Unlike the two above it makes no
      claim about alignment, so it is the one instrument that means
      something on an UNSYNCED seam - i.e. on long_fade, the style every
      other measurement in this repo exempts by name.
"""
import numpy as np

RATE = 44100
FLAM_S = 0.035
CONC_MIN = 1.30         # beat-power concentration below this = no beat
ALIGN_MED_S = 0.030     # median beat delta beyond this = misaligned
ALIGN_HIT = 0.80        # fraction of beats within FLAM_S required
PERC_KICK_HZ = 130.0            # kick fundamental
PERC_BAND_HZ = (1000.0, 5000.0)  # beater click, snare crack, hats
PERC_BIN_S = 0.25
PERC_MASK = 0.25        # clash = quieter kit within ~12 dB of the louder


def _beats_in_span(track, s_lo, s_hi):
    """The track's own beat times (source seconds) inside a span."""
    out = []
    for g in (track.grid or []):
        per = g.get("period_s") or 0
        if per <= 0:
            continue
        t = g["first_beat_s"]
        if t < s_lo:
            t += np.ceil((s_lo - t) / per) * per
        while t <= min(s_hi, g["end_s"]):
            if t >= max(s_lo, g["start_s"]):
                out.append(t)
            t += per
    return np.asarray(out)


def _render_times(beats_src, trace):
    """Project source-time beats into render time via the deck's recorded
    (render_s, source_s) trace - i.e. through the transforms actually
    applied."""
    if len(trace) < 2 or not len(beats_src):
        return np.array([])
    r = np.asarray([t[0] for t in trace])
    s = np.asarray([t[1] for t in trace])
    keep = np.concatenate([[True], np.diff(s) > 1e-6])
    r, s = r[keep], s[keep]
    if len(s) < 2:
        return np.array([])
    inside = (beats_src >= s[0]) & (beats_src <= s[-1])
    return np.interp(beats_src[inside], s, r)


def _low_env(x, sr=RATE):
    from scipy.signal import butter, sosfilt
    sos = butter(4, 120.0, btype="lowpass", fs=sr, output="sos")
    low = sosfilt(sos, x.astype(np.float64))
    env = np.abs(low)
    w = max(int(0.01 * sr), 1)
    return np.convolve(env, np.ones(w) / w, mode="same")


def _concentration(deck_audio, beat_render_ts, lo_s, hi_s):
    """Low-band power on the track's own beats vs everywhere, over
    [lo_s, hi_s] of the render."""
    seg = deck_audio[int(lo_s * RATE):int(hi_s * RATE)].mean(axis=1)
    if len(seg) < RATE:
        return None
    env = _low_env(seg)
    if float(np.mean(env)) < 1e-5:
        return None
    bts = beat_render_ts[(beat_render_ts >= lo_s + 0.05)
                         & (beat_render_ts <= hi_s - 0.05)] - lo_s
    if len(bts) < 8:
        return None
    w = int(0.025 * RATE)
    at = []
    for b in bts:
        i = int(b * RATE)
        at.append(float(np.max(env[max(i - w, 0):i + w])))
    return float(np.mean(at) / (np.mean(env) + 1e-12))


def _transient_env(mono, band, sr=RATE, hop_s=0.005):
    """Positive-difference transient envelope of one band, at 5 ms frames.

    The POSITIVE DIFFERENCE of a smoothed envelope is this codebase's own
    definition of a transient - the submix PLL correlates exactly this
    (deck.py: "the POSITIVE DIFFERENCE of two decks' rings, i.e. their
    TRANSIENTS"), and low-band ENVELOPES were explicitly rejected there
    for the job. Reusing it means this instrument and the PLL agree about
    what a kick is."""
    from scipy.signal import butter, sosfilt
    if isinstance(band, tuple):
        sos = butter(4, list(band), btype="band", fs=sr, output="sos")
    else:
        sos = butter(4, band, btype="lowpass", fs=sr, output="sos")
    y = sosfilt(sos, np.asarray(mono, dtype=np.float64))
    w = max(int(0.010 * sr), 1)
    env = np.convolve(np.abs(y), np.ones(w) / w, mode="same")
    hop = max(int(hop_s * sr), 1)
    n = len(env) // hop * hop
    if n < 2 * hop:
        return np.zeros(0), hop / sr
    frames = env[:n].reshape(-1, hop).max(axis=1)
    return np.maximum(np.diff(frames), 0.0), hop / sr


def _held_bins(t, frame_s, bin_s, hold_s):
    """Per-bin peak transient, held forward across a beat.

    BIN, THEN COMPARE. A transient envelope is spiky - one or two frames
    per hit - so comparing two decks frame-by-frame answers "never
    together" even for two kits hammering over each other.

    HOLD. A kick every 0.5 s makes a deck percussively live for the whole
    half second, not for the 5 ms of the hit. Without a hold a steady
    pattern reads as ~50% duty and the two decks' duty cycles interact,
    so the measurement would track the tempo ratio instead of whether
    both kits are audible. The hold carries the VALUE, not a flag,
    because the comparison below is about relative level."""
    per_bin = max(int(round(bin_s / frame_s)), 1)
    m = len(t) // per_bin * per_bin
    if m < per_bin:
        return np.zeros(0)
    bins = t[:m].reshape(-1, per_bin).max(axis=1)
    out = bins.copy()
    for i in range(1, max(int(round(hold_s / bin_s)), 1)):
        out[i:] = np.maximum(out[i:], bins[:-i])
    return out


def _clash_s(ta, tb, frame_s, t0, t1, bin_s, mask_ratio, hold_s=0.5,
             floor_frac=0.15):
    """Seconds inside [t0, t1] where two percussion beds are BOTH
    audible - i.e. the quieter one is not masked by the louder.

    MASKING, NOT SELF-RELATIVE LEVEL. The obvious rule (each deck against
    its own peak, as mid_overlap_s uses) is wrong here and was measured
    wrong: it works on a blend, where both decks sit near full, but a
    fade's outgoing deck is RECEDING. At 0.25 gain A is below 40% of its
    own solo peak, so a self-relative rule scores it "not playing" while
    it is plainly audible at -12 dB against a quiet incoming track - the
    instrument would excuse exactly the deck whose kick is the problem.

    What decides whether two kicks clash is how far apart they are from
    EACH OTHER: within ~12 dB (mask_ratio 0.25) the quieter one is still
    heard as a competing pulse. `floor_frac` keeps near-silent stretches
    out of it - two inaudible kits are not a clash."""
    if not len(ta) or not len(tb):
        return 0.0
    ha = _held_bins(ta, frame_s, bin_s, hold_s)
    hb = _held_bins(tb, frame_s, bin_s, hold_s)
    n = min(len(ha), len(hb))
    if n == 0:
        return 0.0
    ha, hb = ha[:n], hb[:n]
    louder = np.maximum(ha, hb)
    quieter = np.minimum(ha, hb)
    ref = float(np.percentile(louder, 95))
    if ref <= 1e-9:
        return 0.0
    idx = np.arange(n) * bin_s
    clash = ((quieter > mask_ratio * louder) & (louder > floor_frac * ref)
             & (idx >= t0) & (idx <= t1))
    return float(clash.sum() * bin_s)


def perc_overlap(deck_a, deck_b, t0, t1, bin_s=PERC_BIN_S,
                 mask_ratio=PERC_MASK):
    """How long two decks are percussively live AT THE SAME TIME, in
    seconds, over the render window [t0, t1].

    `deck_a`/`deck_b` are post-EQ, post-gain per-deck renders (n, ch) -
    they must be post-EQ or the measurement cannot see a band carve, and
    post-gain or it cannot see the fader.

    Returns {"kick_s", "perc_s"}: co-presence below 130 Hz (two kick
    drums) and in 1-5 kHz (beater click, snare, hats). Kept SEPARATE
    because they have separate cures - a low-band handover fixes the
    first, only a shelf or a different in-point touches the second.

    This is the missing instrument for long_fade. A fade emits no `sync`
    event, so max_err_beats is 0.000 on every fade ever logged and the
    live assessor can only ever call it clean or hole; kick-to-kick
    distance is meaningless on decks that were never matched. Co-presence
    is the thing that is actually wrong, and it needs no shared grid."""
    out = {"kick_s": 0.0, "perc_s": 0.0}
    for key, band in (("kick_s", PERC_KICK_HZ), ("perc_s", PERC_BAND_HZ)):
        env = []
        for d in (deck_a, deck_b):
            x = np.asarray(d, dtype=np.float64)
            if x.ndim > 1:
                x = x.mean(axis=1)
            env.append(_transient_env(x, band))
        (ta, fs_a), (tb, _fs_b) = env
        out[key] = round(_clash_s(ta, tb, fs_a, t0, t1, bin_s,
                                  mask_ratio), 2)
    return out


def verify_seam(decks, marks, a_track, b_track):
    """(ok, report). decks: {"a","b"} post-EQ renders; marks carries
    blend_s/swap_s and the per-deck position traces."""
    rep = {}
    tr_a, tr_b = marks["pos"]["a"], marks["pos"]["b"]
    blend, swap = marks["blend_s"], marks["swap_s"]
    dual_hi = min(swap, blend + 16.0)

    # -- beats of each track, in render time, through the real transforms
    def beats_r(track, trace):
        if not trace:
            return np.array([])
        src_lo = min(s for _r, s in trace)
        src_hi = max(s for _r, s in trace)
        return _render_times(_beats_in_span(track, src_lo, src_hi), trace)
    ba = beats_r(a_track, tr_a)
    bb = beats_r(b_track, tr_b)

    # -- GRID HONESTY on each deck's solo span ("where the power would be")
    ca = _concentration(decks["a"], ba, max(blend - 12.0, 0.0), blend)
    n_r = len(decks["b"]) / RATE
    cb = _concentration(decks["b"], bb, min(swap + 1.0, n_r),
                        min(swap + 13.0, n_r))
    rep["conc_a"], rep["conc_b"] = (round(ca, 2) if ca else None,
                                    round(cb, 2) if cb else None)
    for side, c in (("a", ca), ("b", cb)):
        if c is not None and c < CONC_MIN:
            rep["verdict"] = f"no_beat_power_{side}"
            return False, rep

    # -- DECK ALIGNMENT across the dual span
    da = ba[(ba >= blend) & (ba <= dual_hi)]
    db_ = bb[(bb >= blend) & (bb <= dual_hi)]
    if len(da) < 4 or len(db_) < 4:
        rep["verdict"] = "no_overlap_beats"
        return True, rep
    deltas = np.array([float(np.min(np.abs(db_ - t))) for t in da])
    rep["align_med_ms"] = round(float(np.median(deltas)) * 1000, 1)
    rep["align_hit"] = round(float(np.mean(deltas <= FLAM_S)), 2)
    if np.median(deltas) > ALIGN_MED_S or rep["align_hit"] < ALIGN_HIT:
        rep["verdict"] = "decks_misaligned"
        return False, rep
    rep["verdict"] = "ok"
    return True, rep
