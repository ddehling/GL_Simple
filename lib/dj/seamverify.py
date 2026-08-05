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
"""
import numpy as np

RATE = 44100
FLAM_S = 0.035
CONC_MIN = 1.30         # beat-power concentration below this = no beat
ALIGN_MED_S = 0.030     # median beat delta beyond this = misaligned
ALIGN_HIT = 0.80        # fraction of beats within FLAM_S required


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
