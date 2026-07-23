"""Beat-synchronous RHYTHM SIGNATURES + pairwise beat-compatibility.

Key (Camelot/chroma) answers "will these two tracks clash melodically";
this module answers the other half: "will their GROOVES lock". Per track
we store a compact signature (DB v13 tracks.rhythm) computed by folding
band-split onset flux onto the beat grid:

    low/mid/high    2-bar x 16th-note step patterns (32 steps each) -
                    where the kicks / snares / hats actually land
    beat_low/perc   one-beat fine folds (36 bins, ~14ms at 120bpm) for
                    swing measurement and flam simulation
    swing           offbeat position within the beat: 0.50 straight,
                    0.67 full triplet shuffle
    density         fraction of active steps (busy break vs sparse 4x4)

The signature is TEMPO-INVARIANT (everything is beat-phase folded), so two
tracks compare correctly at any meet tempo, including half/double-time
reads (the pattern is resampled by the tempo multiple).

seam_rhythm() turns two signatures + a planned stretch rate into the
decomposed seam terms the planner surfaces (kick agreement, swing delta,
flam risk, pattern similarity) plus one bounded 0..1 score the brain can
fold into selection. Every term is EVIDENCE-GATED: missing signatures ->
None, and the caller treats that as neutral (no evidence, no penalty).

Computed from the htdemucs DRUM STEM when one is on disk (clean rhythm
section, no melodic transients polluting the fold), else from the full
mix's band-split flux. Pure numpy, no torch, multiprocessing-safe.
"""
import math

import numpy as np

SIG_VERSION = 2                  # v2: in/out REGION patterns + meter
STEPS_PER_BEAT = 4               # 16th-note grid
PATTERN_BARS = 2                 # folded over 2 bars -> 32 steps
N_STEPS = STEPS_PER_BEAT * 4 * PATTERN_BARS
BEAT_BINS = 36                   # one-beat fine fold (~14ms/bin at 120bpm)
REGION_S = 48.0                  # in/out region window (~24 bars at 120bpm)
_REGIONS = ("in", "out")
_REGION_KEYS = tuple(f"{r}_{n}" for r in _REGIONS
                     for n in ("low", "mid", "high"))

# Swing search window inside the beat: the offbeat 8th lives between
# phase 0.40 (rushed) and 0.80 (hard shuffle); 0.50 = straight.
_SWING_LO, _SWING_HI = 0.40, 0.80

# Half/double(+polymetric) tempo reads the mixing engine actually uses.
KNOWN_MULTS = (1.0, 2.0, 0.5, 0.75, 1.5)


# --------------------------------------------------------------------------
# Signature extraction
# --------------------------------------------------------------------------

def _band_flux(bands):
    """Positive spectral flux per band group: low (kick/bass, bands 0-6),
    mid (snare/clap, 6-16), high (hats, 16-32). Same normalization as
    features._onset_channels so mix-derived and stem-derived signatures
    live on comparable scales."""
    mean = np.maximum(bands.mean(axis=0, keepdims=True), 1e-10)
    nb = bands / mean
    flux = np.maximum(0.0, np.diff(nb, axis=0, prepend=nb[:1]))
    return (flux[:, 0:6].mean(axis=1), flux[:, 6:16].mean(axis=1),
            flux[:, 16:32].mean(axis=1))


def _main_segment(grid):
    if not grid:
        return None
    return max(grid, key=lambda s: s["end_s"] - s["start_s"])


def _fold(flux, times, origin_s, span_beats, period_s, n_bins):
    """Mean flux per phase bin, folding [origin + k*span] onto one span."""
    phases = ((times - origin_s) / (period_s * span_beats)) % 1.0
    bins = np.minimum((phases * n_bins).astype(np.int64), n_bins - 1)
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    np.add.at(acc, bins, flux)
    np.add.at(cnt, bins, 1)
    return acc / np.maximum(cnt, 1)


def _norm_pattern(p):
    """0..1 pattern + a peakedness weight (how rhythmic the channel is).
    A flat channel (pads, near-silence) normalizes to noise - its weight
    goes to ~0 so scoring ignores it."""
    base = float(np.median(p))
    peak = float(p.max())
    if peak <= 1e-9:
        return np.zeros_like(p), 0.0
    out = np.clip((p - base) / max(peak - base, 1e-9), 0.0, 1.0)
    # Peakedness: how far the top steps stand off the median floor.
    w = float(np.clip((peak / max(base, 1e-9) - 1.0) / 4.0, 0.0, 1.0))
    return out, w


def _measure_swing(beat_fold):
    """Offbeat-8th position within the beat from a one-beat fine fold.
    Returns (swing 0.40..0.80, conf 0..1); (0.5, 0.0) when there is no
    discernible offbeat onset (nothing to disagree about)."""
    n = len(beat_fold)
    lo, hi = int(_SWING_LO * n), int(_SWING_HI * n)
    if hi - lo < 3:
        return 0.5, 0.0
    win = beat_fold[lo:hi]
    k = int(np.argmax(win))
    peak = float(win[k])
    base = float(np.median(beat_fold))
    span = float(beat_fold.max()) - base
    if span <= 1e-9 or (peak - base) / span < 0.25:
        return 0.5, 0.0                     # no real offbeat hit
    pos = lo + k
    if 0 < k < len(win) - 1:                # parabolic sub-bin
        y0, y1, y2 = win[k - 1], win[k], win[k + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            pos += 0.5 * (y0 - y2) / den
    swing = float(np.clip(pos / n, _SWING_LO, _SWING_HI))
    # Confidence = how far the offbeat peak stands off its own floor, NOT
    # off the fold's global max: the downbeat (kick+snare) always dwarfs
    # the hats, and normalizing by it scaled real swing clashes (0.13
    # delta, measured) under the warning threshold.
    conf = float(np.clip((peak - base) / max(peak + base, 1e-9), 0.0, 1.0))
    return swing, conf


def _step_patterns(fluxes, times, sel, bar0, period):
    """low/mid/high step patterns + peakedness weights for one frame
    selection (a boolean mask or slice over the segment)."""
    pats, weights = {}, {}
    for name, flux in fluxes:
        folded = _fold(flux[sel], times[sel], bar0, 4 * PATTERN_BARS,
                       period, N_STEPS)
        pat, w = _norm_pattern(folded)
        pats[name] = pat
        weights[name] = w
    return pats, weights


def _detect_meter(accent, times, bar0, period):
    """3/4 vs 4/4 from bar-profile peakedness: fold the accent flux at
    3-beat and 4-beat bars; the true meter's fold keeps its accents
    aligned (high contrast), the wrong one smears them flat. Returns
    (meter 3|4, conf 0..1); defaults to 4 - the claim '3/4' needs a
    strong margin because nearly everything in a club library is 4/4."""
    def contrast(nbeats):
        # ONE bin per beat: the discriminator is the bar's ACCENT profile
        # (which beats are heavy), and 16th-resolution bins let the empty
        # space between beats dominate the contrast of both candidate
        # meters equally (measured: a synthetic waltz read 4/4).
        prof = _fold(accent, times, bar0, nbeats, period, nbeats)
        m = prof.mean()
        return float(prof.std() / m) if m > 1e-9 else 0.0
    c3, c4 = contrast(3), contrast(4)
    if c3 > c4 * 1.3 and c3 > 0.2:
        conf = float(np.clip((c3 / max(c4, 1e-9) - 1.3) / 0.7, 0.0, 1.0))
        return 3, conf
    conf = float(np.clip((c4 / max(c3, 1e-9) - 1.0) / 0.6, 0.0, 1.0))
    return 4, conf


def rhythm_signature(bands, grid, downbeat_offset, fps=40, source="mix",
                     latency_s=0.028, mix_in_s=None, mix_out_s=None):
    """Fold a framed track (features.frame_track band matrix) onto its beat
    grid -> the JSON-able signature dict, or None when the grid is unusable.

    `bands` may come from the full mix OR from a decoded drum stem framed
    through the same pipeline (the backfill tool's preferred path).
    `latency_s` is features.ONSET_LATENCY_S: spectral-flux peaks LEAD the
    true transient, and the grid is latency-corrected while raw flux is
    not - without compensating, straight 4x4 tracks measured swing ~0.44
    instead of 0.50 (28ms early at 120bpm, seen on the real library).

    v2: when `mix_in_s` / `mix_out_s` (the track's primary entry/exit
    points) are given, REGION patterns are folded over ~REGION_S-second
    windows there too - a blend compares A's out-region against B's
    in-region, not two whole-track averages (intros are routinely sparser
    than the groove body). Also detects 3/4 vs 4/4 (meter)."""
    g = _main_segment(grid)
    if g is None or g["period_s"] <= 0:
        return None
    period = g["period_s"]
    span_s = g["end_s"] - g["start_s"]
    if span_s < period * 4 * PATTERN_BARS * 2:      # < 4 bars of evidence
        return None
    low, mid, high = _band_flux(bands)
    f0 = int(max(g["start_s"], 0.0) * fps)
    f1 = min(int(g["end_s"] * fps), len(low))
    if f1 - f0 < fps * 8:
        return None
    times = np.arange(f0, f1) / fps + latency_s
    bar0 = g["first_beat_s"] + (downbeat_offset or 0) * period
    fluxes = (("low", low[f0:f1]), ("mid", mid[f0:f1]),
              ("high", high[f0:f1]))

    pats, weights = _step_patterns(fluxes, times, slice(None), bar0, period)
    meter, meter_conf = _detect_meter(
        low[f0:f1] + mid[f0:f1], times, bar0, period)

    beat_low = _fold(low[f0:f1], times, g["first_beat_s"], 1, period,
                     BEAT_BINS)
    beat_perc = _fold(mid[f0:f1] + high[f0:f1], times, g["first_beat_s"],
                      1, period, BEAT_BINS)
    bl, _ = _norm_pattern(beat_low)
    bp, _ = _norm_pattern(beat_perc)
    swing, swing_conf = _measure_swing(bp)

    # Density: active 16th steps across the channels that actually carry
    # rhythm, weighted by channel peakedness.
    wsum = sum(weights.values())
    if wsum > 1e-6:
        density = sum(float((pats[n] > 0.3).mean()) * weights[n]
                      for n in pats) / wsum
    else:
        density = 0.0

    out = {
        "v": SIG_VERSION, "steps_per_beat": STEPS_PER_BEAT,
        "bars": PATTERN_BARS, "source": source,
        "low": [round(float(x), 3) for x in pats["low"]],
        "mid": [round(float(x), 3) for x in pats["mid"]],
        "high": [round(float(x), 3) for x in pats["high"]],
        "w_low": round(weights["low"], 3),
        "w_mid": round(weights["mid"], 3),
        "w_high": round(weights["high"], 3),
        "beat_low": [round(float(x), 3) for x in bl],
        "beat_perc": [round(float(x), 3) for x in bp],
        "swing": round(swing, 4), "swing_conf": round(swing_conf, 3),
        "density": round(float(density), 3),
        "meter": meter, "meter_conf": round(meter_conf, 3),
    }

    # Region patterns around the primary entry/exit points.
    for rname, anchor in (("in", mix_in_s), ("out", mix_out_s)):
        if anchor is None:
            continue
        if rname == "in":
            r0, r1 = float(anchor), float(anchor) + REGION_S
        else:
            r0, r1 = float(anchor) - REGION_S, float(anchor)
        sel = (times - latency_s >= r0) & (times - latency_s < r1)
        if sel.sum() < fps * 16:         # < 16s of evidence in the window
            continue
        rpats, _rw = _step_patterns(fluxes, times, sel, bar0, period)
        for n in ("low", "mid", "high"):
            out[f"{rname}_{n}"] = [round(float(x), 3) for x in rpats[n]]
    return out


def prep_signature(blob):
    """Hydrate a stored signature: lists -> numpy arrays, once, at library
    load (seam scoring touches these per candidate). None-safe."""
    if not blob or not isinstance(blob, dict) or "low" not in blob:
        return None
    sig = dict(blob)
    try:
        for k in ("low", "mid", "high", "beat_low", "beat_perc"):
            sig[k] = np.asarray(blob[k], dtype=np.float64)
        for k in _REGION_KEYS:               # v2, optional
            if blob.get(k) is not None:
                sig[k] = np.asarray(blob[k], dtype=np.float64)
        if len(sig["low"]) != N_STEPS:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return sig


def region_view(sig, region):
    """The signature as one side of a seam hears it: low/mid/high replaced
    by that region's patterns when the v2 signature has them ('out' for
    the exiting track, 'in' for the entering one). Fine folds, swing and
    weights stay global - microtiming doesn't change between sections the
    way patterns do. Falls back to the whole-track view (v1, or a region
    window that had too little evidence)."""
    if sig is None:
        return None
    keys = {n: f"{region}_{n}" for n in ("low", "mid", "high")}
    if not all(sig.get(k) is not None for k in keys.values()):
        return sig
    view = dict(sig)
    for n, k in keys.items():
        view[n] = sig[k]
    view["region"] = region
    return view


# --------------------------------------------------------------------------
# Pairwise terms
# --------------------------------------------------------------------------

def _resample_pattern(pat, mult):
    """B's step pattern as A hears it when B is read at tempo-multiple
    `mult` (eff_bpm = bpm_b * mult). mult=2: B's 8ths count as beats, so
    A-step j samples B at j/2; mult=0.5 samples at 2j (wrapping)."""
    if abs(mult - 1.0) < 1e-6:
        return pat
    n = len(pat)
    idx = (np.arange(n) / mult) % n
    i0 = idx.astype(np.int64)
    fr = idx - i0
    return pat[i0] * (1 - fr) + pat[(i0 + 1) % n] * fr


def _wcos(a, b):
    """Cosine similarity weighted toward the steps where anything happens
    (silent agreement is easy; agreeing on the hits is what matters)."""
    w = np.maximum(a, b)
    if w.sum() <= 1e-9:
        return 1.0
    aw, bw = a * np.sqrt(w), b * np.sqrt(w)
    den = np.linalg.norm(aw) * np.linalg.norm(bw)
    if den <= 1e-12:
        return 1.0
    return float(np.dot(aw, bw) / den)


def _best_rotation_sim(a, b, rotations):
    """Max weighted-cosine over the given step rotations (bar ambiguity
    under half/double reads: which of A's bars aligns with B's is not
    determined by the phrase-aligned blend)."""
    best, best_rot = -1.0, 0
    for r in rotations:
        s = _wcos(a, np.roll(b, r))
        if s > best:
            best, best_rot = s, r
    return best, best_rot


def _peak_phases(fold, thr=0.5):
    """Phases (0..1) of local maxima above thr in a one-beat fold."""
    n = len(fold)
    out = []
    for i in range(n):
        v = fold[i]
        if v >= thr and v >= fold[i - 1] and v >= fold[(i + 1) % n]:
            out.append(i / n)
    return out


def _flam_ms(sig_a, sig_b, mult, period_s):
    """Closest near-coincident low/perc hit pair across the two beat folds,
    in ms at the meet tempo. Hits 20-80ms apart read as a machine-gun flam;
    <15ms fuses, >90ms reads as intentional syncopation. None when either
    side has no confident peaks. Tempo-multiple reads compare at the
    EFFECTIVE beat (B's fold is phase-invariant to 2x/0.5x by symmetry of
    its own grid - close enough for a risk estimate)."""
    pa = _peak_phases(sig_a["beat_low"]) + _peak_phases(sig_a["beat_perc"])
    pb = _peak_phases(sig_b["beat_low"]) + _peak_phases(sig_b["beat_perc"])
    if not pa or not pb:
        return None
    best = 1.0
    for x in pa:
        for y in pb:
            d = abs(x - y)
            d = min(d, 1.0 - d)
            best = min(best, d)
    return round(best * period_s * 1000.0, 1)


def rhythm_terms(sig_a, sig_b, mult=1.0, period_s=0.5):
    """Decomposed beat-compatibility between two prepared signatures, with
    B read at tempo-multiple `mult` and the blend running at beat period
    `period_s` (seconds, meet tempo). Returns a dict of terms + a bounded
    composite `score` 0..1, or None when either signature is missing."""
    if sig_a is None or sig_b is None:
        return None
    steps_per_bar = STEPS_PER_BEAT * 4
    # Downbeats are aligned by the blend; under a tempo-multiple read the
    # BAR correspondence is ambiguous -> try bar-shifts of the resampled
    # pattern (and half-bar for the polymetric reads).
    rotations = (0,) if abs(mult - 1.0) < 1e-6 else \
        (0, steps_per_bar, steps_per_bar // 2,
         -steps_per_bar // 2)

    b_low = _resample_pattern(sig_b["low"], mult)
    kick_agr, rot = _best_rotation_sim(sig_a["low"], b_low, rotations)
    sims, wsum = 0.0, 0.0
    for name in ("mid", "high"):
        w = min(sig_a.get(f"w_{name}", 0.0), sig_b.get(f"w_{name}", 0.0))
        if w <= 0.05:
            continue
        s = _wcos(sig_a[name],
                  np.roll(_resample_pattern(sig_b[name], mult), rot))
        sims += w * s
        wsum += w
    pattern_sim = sims / wsum if wsum > 1e-6 else None

    conf = min(sig_a.get("swing_conf", 0.0), sig_b.get("swing_conf", 0.0))
    swing_delta = abs(sig_a.get("swing", 0.5) - sig_b.get("swing", 0.5))
    swing_delta *= conf                      # unmeasured swing can't clash
    dens_a = sig_a.get("density", 0.0)
    dens_b = sig_b.get("density", 0.0)
    flam = _flam_ms(sig_a, sig_b, mult, period_s)

    # Composite: kick agreement dominates (the audible train-wreck term),
    # swing clash second (nothing fixes it), then flam-band near-misses,
    # percussion pattern and density comparability. Each sub-term already
    # lives in 0..1.
    s_kick = max(kick_agr, 0.0)
    s_swing = math.exp(-((swing_delta / 0.06) ** 2))
    s_pat = 0.5 + 0.5 * max(pattern_sim, 0.0) if pattern_sim is not None \
        else 0.75
    s_dens = 1.0 - 0.5 * abs(dens_a - dens_b)
    # Flam dip: hits ~40ms apart are the machine-gun zone; <15ms fuses,
    # >90ms reads as intentional syncopation. No confident peaks = neutral.
    s_flam = 1.0
    if flam is not None:
        s_flam = 1.0 - 0.75 * math.exp(-0.5 * ((flam - 40.0) / 18.0) ** 2)
    score = float(np.clip(
        0.40 * s_kick + 0.22 * s_swing + 0.15 * s_pat + 0.10 * s_dens
        + 0.13 * s_flam, 0.0, 1.0))
    # METER GATE: 3/4 against 4/4 fails regardless of everything above.
    # Only the side CLAIMING 3/4 must be confident - claiming 3 already
    # took a strong margin, while "4" is the default state (real 4x4
    # tracks with kick-every-beat accents sit near-flat and never reach
    # high 4-confidence; v1 signatures default to 4 and never trigger).
    meter_a = sig_a.get("meter", 4)
    meter_b = sig_b.get("meter", 4)
    conf3 = (sig_a if meter_a == 3 else sig_b).get("meter_conf", 0.0)
    meter_clash = meter_a != meter_b and conf3 >= 0.4
    if meter_clash:
        score *= 0.3
    return {
        "score": round(score, 3),
        "kick_agreement": round(float(kick_agr), 3),
        "pattern_sim": (round(float(pattern_sim), 3)
                        if pattern_sim is not None else None),
        "swing_a": sig_a.get("swing", 0.5),
        "swing_b": sig_b.get("swing", 0.5),
        "swing_delta": round(float(swing_delta), 4),
        "swing_conf": round(float(conf), 3),
        "density_a": dens_a, "density_b": dens_b,
        "flam_ms": flam,
        "meter_a": meter_a, "meter_b": meter_b,
        "meter_clash": bool(meter_clash),
        "regions": f"{sig_a.get('region', 'full')}/"
                   f"{sig_b.get('region', 'full')}",
        "rot": int(rot), "mult": mult,
        "source": f"{sig_a.get('source', '?')}/{sig_b.get('source', '?')}",
    }


def aligned_pattern(sig, name, mult=1.0, rot=0):
    """A signature channel as the other deck hears it: resampled by the
    tempo-multiple read and rotated by the bar alignment rhythm_terms
    chose. What the seam inspector draws so the two grids line up the way
    the blend will actually sound."""
    if sig is None or name not in sig:
        return None
    return np.roll(_resample_pattern(sig[name], mult), rot)


def tempo_mult_for(bpm_a, bpm_b, rate):
    """The tempo-multiple B was read at, recovered from a planned stretch
    rate (rate = bpm_a / (bpm_b * mult)) and snapped to the known reads."""
    if not bpm_a or not bpm_b or not rate:
        return 1.0
    raw = bpm_a / (rate * bpm_b)
    return min(KNOWN_MULTS, key=lambda m: abs(math.log(max(raw, 1e-6) / m)))


_MULT_LABEL = {2.0: "2x-time", 0.5: "half-time", 0.75: "3:4 read",
               1.5: "3:2 read"}


def seam_chips(plan, seam_info):
    """Word-first seam badges: the LIMITING FACTOR of a seam as a short
    label, not a number dump. Empty list = rhythmically clean (silence is
    information). Shared by the plan list, the arc-strip tooltips and the
    copilot; a trailing '?' marks terms built on low-confidence grids
    ("the estimate is shaky" is different information than "the seam is
    bad")."""
    chips = []
    si = seam_info or {}
    rt = si.get("rhythm") or {}
    q = "?" if rt and rt.get("conf", 1.0) < 0.5 else ""
    mult = si.get("mult", rt.get("mult"))
    if mult and abs(mult - 1.0) > 1e-6:
        chips.append(_MULT_LABEL.get(mult, f"{mult:g}x read"))
    rate = (plan or {}).get("rate")
    if rate and abs(rate - 1.0) > 0.04:            # audibility knee
        chips.append(f"stretch {(rate - 1) * 100:+.1f}%")
    if rt:
        if rt.get("meter_clash"):
            chips.append(f"{rt.get('meter_a', 4)}/4 vs "
                         f"{rt.get('meter_b', 4)}/4" + q)
        if rt.get("swing_delta", 0.0) > 0.055:
            a, b = rt.get("swing_a", 0.5), rt.get("swing_b", 0.5)
            chips.append(("swung vs straight" if a > b
                          else "straight vs swung") + q)
        if rt.get("kick_agreement", 1.0) < 0.35:
            chips.append("kick clash" + q)
        fl = rt.get("flam_ms")
        if fl is not None and 15.0 <= fl <= 80.0:
            chips.append(f"flam risk {fl:.0f}ms" + q)
    return chips


def seam_rhythm(a, b, rate=1.0):
    """The planner/compiler entry point: rhythm terms for a seam a -> b at
    the planned stretch rate, comparing A's OUT-region pattern against B's
    IN-region pattern when the v2 signatures carry them (the material the
    blend actually overlaps - intros are routinely sparser than the body).
    Reads .rhythm_sig / .bpm / .bpm_conf off the TrackInfo-likes; None
    when either track lacks a signature."""
    sig_a = region_view(getattr(a, "rhythm_sig", None), "out")
    sig_b = region_view(getattr(b, "rhythm_sig", None), "in")
    if sig_a is None or sig_b is None:
        return None
    mult = tempo_mult_for(getattr(a, "bpm", 0.0), getattr(b, "bpm", 0.0),
                          rate)
    period = 60.0 / a.bpm if getattr(a, "bpm", 0.0) else 0.5
    rt = rhythm_terms(sig_a, sig_b, mult=mult, period_s=period)
    if rt is None:
        return None
    # Estimate trust: shaky grids make every term above shaky. Surfaced so
    # the UI can mark chips with '?' instead of asserting them.
    rt["conf"] = round(min(getattr(a, "bpm_conf", 1.0) or 0.0,
                           getattr(b, "bpm_conf", 1.0) or 0.0), 2)
    return rt
