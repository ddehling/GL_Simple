"""Predicted-exposure dip scan: keep blends from lurching over a
track's stop-gap.

Born 2026-08-17 from the night census's lurch anatomy. The worst
long_blend lurches were HUSH-THEN-SLAM events: a sub-second structural
silence in one track (a pre-drop stop, an outro breakdown edge)
landing exactly where the automation had left that deck carrying the
room - the mix drops ~15 dB for half a second and slams back. Section
labels miss them (too short) and the 2 Hz energy curve is too smoothed.

A raw PCM hole scan was measured NOT to discriminate (holes appear at
the same rate in healthy blends - the other deck masks them). What
discriminates is EXPOSURE: hole x automation. Both are knowable at
plan time - the PCM of both tracks is already in RAM (live: A on its
deck, B predecoded; offline: the harness decodes both) and the event
schedule is deterministic - so this module PREDICTS the blend's level
envelope from windowed band-RMS + the compiled gain/EQ automation,
finds hush-then-slam dips, and adjusts the plan: end the blend before
a late dip (shrink beats - the established exposure lever) or concede
an early one to the fade, whose dip is deliberate.

Scoped to the plain blends (long_blend / bass_swap / filter_sweep) -
the measured lurch class (43 of 46 census specimens). Stem styles play
partial stems the full-mix RMS can't model; they keep their geometry.
"""
import numpy as np

RATE = 44100
DT = 0.25               # prediction grid / RMS window (s)
DIP_DB = -12.0          # dip = this far under the blend's own median...
SLAM_DB = 8.0           # ...that RECOVERS this much within RECOVER_S
RECOVER_S = 1.5         # (hush-then-slam; a fade-out tail never recovers)
EDGE_S = 2.0            # ignore the blend's first seconds (B's entry ramp)
GAP_STYLES = ("long_blend", "bass_swap", "filter_sweep")


def band_rms(samples, lo_s, hi_s, rate=RATE):
    """(low, rest) windowed RMS at DT for [lo_s, hi_s) of a track.
    low < 130 Hz (kick/bass fundamentals), rest = everything else."""
    from scipy.signal import butter, sosfilt
    x = np.asarray(samples)
    if x.ndim > 1:
        x = x.mean(axis=1)
    i0, i1 = max(int(lo_s * rate), 0), min(int(hi_s * rate), len(x))
    seg = x[i0:i1].astype(np.float64)
    w = int(DT * rate)
    if len(seg) < 2 * w:
        return np.zeros(0), np.zeros(0)
    sos = butter(4, 130.0, fs=rate, output="sos")
    lo = sosfilt(sos, seg)
    rest = seg - lo
    n = len(seg) // w * w

    def r(y):
        return np.sqrt(np.mean(y[:n].reshape(-1, w) ** 2, axis=1))
    return r(lo), r(rest)


def _envelopes(events, deck, t_grid, S0, rate=RATE):
    """(gain, eq_low, eq_rest) envelopes for one deck over t_grid
    (seconds rel S0), replayed from the event schedule with the same
    overwrite semantics the decks use (a new ramp replaces a pending
    one). Constant-power gain curves interpolate in the power domain."""
    gain = np.ones(len(t_grid))
    eqlo = np.ones(len(t_grid))
    mid_env = np.ones(len(t_grid))
    high_env = np.ones(len(t_grid))
    evs = sorted((e for e in events
                  if e.get("deck") == deck
                  and e.get("cmd") in ("gain", "eq")),
                 key=lambda e: e["at"])
    # Replay with the decks' overwrite semantics: a new ramp starts from
    # the value the old one had reached, and replaces it.
    chans = ("gain", "low", "mid", "high")
    cur_v = {k: 1.0 for k in chans}
    ramp_end = {k: -1e9 for k in chans}
    ramp_from = dict(cur_v)
    ramp_tgt = dict(cur_v)
    ramp_start = {k: -1e9 for k in chans}
    ramp_curve = {k: "linear" for k in chans}

    def value_at(k, t):
        if t >= ramp_end[k]:
            return ramp_tgt[k]
        if t <= ramp_start[k]:
            return ramp_from[k]
        u = (t - ramp_start[k]) / max(ramp_end[k] - ramp_start[k], 1e-6)
        v0, v1 = ramp_from[k], ramp_tgt[k]
        if ramp_curve[k] == "power":
            return float(np.sqrt(v0 * v0 + (v1 * v1 - v0 * v0) * u))
        return v0 + (v1 - v0) * u

    def start_ramp(k, t, target, ramp_s, curve="linear"):
        ramp_from[k] = value_at(k, t)
        ramp_tgt[k] = float(target)
        ramp_start[k] = t
        ramp_end[k] = t + max(float(ramp_s or 0.0), 1e-3)
        ramp_curve[k] = curve

    for e in evs:
        t = (e["at"] - S0) / rate
        if e["cmd"] == "gain":
            start_ramp("gain", t, e.get("value", 1.0),
                       e.get("ramp_s", 0.05), e.get("curve", "linear"))
        else:
            for band in ("low", "mid", "high"):
                if e.get(band) is not None:
                    start_ramp(band, t, e[band], e.get("ramp_s", 0.05))
    for i, t in enumerate(t_grid):
        gain[i] = value_at("gain", t)
        eqlo[i] = value_at("low", t)
        mid_env[i] = value_at("mid", t)
        high_env[i] = value_at("high", t)
    return gain, eqlo, (mid_env + high_env) / 2.0


def predicted_dips(events, plan, cur, cand, samples_a, samples_b,
                   S0, end, rate=RATE):
    """Hush-then-slam dips the automation will EXPOSE, predicted from
    band RMS x replayed envelopes. Returns [(t_rel_s, depth_db)]."""
    span = (end - S0) / rate
    t_grid = np.arange(0.0, span, DT)
    if len(t_grid) < 8:
        return []
    ra = plan.get("a_rate") or 1.0
    rb = plan.get("rate") or 1.0
    la, ra_ = band_rms(samples_a, plan["out_s"], plan["out_s"] + span * ra)
    lb, rb_ = band_rms(samples_b, plan["in_s"], plan["in_s"] + span * rb)
    n = min(len(t_grid), len(la), len(lb))
    if n < 8:
        return []
    t_grid = t_grid[:n]
    ga, elo_a, erest_a = _envelopes(events, "a", t_grid, S0, rate)
    gb, elo_b, erest_b = _envelopes(events, "b", t_grid, S0, rate)
    # filter_sweep's A-side HP sweep isn't an eq event - approximate:
    # A loses its low and half its rest over the back half.
    if plan.get("style") == "filter_sweep":
        back = t_grid >= span * 0.5
        u = np.clip((t_grid - span * 0.5) / max(span * 0.5, 1e-6), 0, 1)
        elo_a = elo_a * np.where(back, 1.0 - u, 1.0)
        erest_a = erest_a * np.where(back, 1.0 - 0.5 * u, 1.0)
    pa = (ga ** 2) * ((elo_a ** 2) * (la[:n] ** 2)
                      + (erest_a ** 2) * (ra_[:n] ** 2))
    pb = (gb ** 2) * ((elo_b ** 2) * (lb[:n] ** 2)
                      + (erest_b ** 2) * (rb_[:n] ** 2))
    level = np.sqrt(pa + pb)
    ref = float(np.median(level[level > 1e-7]))
    if ref < 1e-6:
        return []
    db = 20 * np.log10(np.maximum(level, 1e-7) / ref)
    dips = []
    k = int(RECOVER_S / DT)
    for i in range(int(EDGE_S / DT), n):
        if db[i] <= DIP_DB:
            ahead = db[i + 1:i + 1 + k]
            if len(ahead) and ahead.max() >= db[i] + SLAM_DB:
                if not dips or t_grid[i] - dips[-1][0] > 1.0:
                    dips.append((float(t_grid[i]), round(float(db[i]), 1)))
    return dips


def apply_gap_policy(brain, plan, cur, cand, meta, samples_a, samples_b,
                     snapshot, active, incoming, after_s=None):
    """Compile, predict, and if the blend would lurch over a hush,
    adjust: end it before a late dip (shrink beats) or concede an early
    one to the fade. Returns (plan, events, swap_at, blend_at, action) -
    events are the FINAL compile. One implementation shared by the live
    arm path and the offline harness, or the simulator stops predicting
    the night.

    The action lands on plan['gap'] so night logs and the census can
    count it."""
    events, swap_at, blend_at = brain.build_events(
        plan, snapshot, active, incoming, cur, cand)
    # Urgent (skip/mix-now) seams want speed, not analysis - and their
    # compressed geometry rarely reaches the late-dip zone anyway.
    if plan.get("style") not in GAP_STYLES or plan.get("urgent") \
            or samples_a is None or samples_b is None:
        return plan, events, swap_at, blend_at, None
    S0 = blend_at
    beats = int(plan.get("beats", 32) or 32)
    end = S0 + int(beats * cur.period_s / max(plan.get("a_rate") or 1.0,
                                              1e-6) * RATE)
    try:
        dips = predicted_dips(events, plan, cur, cand,
                              samples_a, samples_b, S0, end)
    except Exception:
        return plan, events, swap_at, blend_at, None
    # POSITION IS THE DISCRIMINATOR (measured on the census specimens):
    # late dips (final ~third of the blend, where one deck carries the
    # room alone) are the hush-then-slam lurches - 5+ of the census's
    # worst-10 - while EARLY predicted dips appear at the same rate in
    # healthy blends (B's entry machinery covers them) and never
    # measured as lurch. Act only on the late class: 1/51 healthy
    # blends is touched, and that one merely ends a phrase early.
    span_s = (end - S0) / RATE
    late = [d for d in dips if d[0] >= 0.6 * span_s]
    if not late:
        return plan, events, swap_at, blend_at, None
    first = late[0][0]
    safe_beats = int((first - 2.0) / max(cur.period_s, 1e-6))
    safe_beats = (safe_beats // 8) * 8
    if safe_beats >= 16:
        plan["beats"] = min(beats, safe_beats)
        plan["gap"] = {"at_rel_s": round(first, 2),
                       "depth_db": late[0][1], "action": "shortened",
                       "beats": plan["beats"]}
    else:
        old = plan
        plan = brain.plan_transition(cur, cand, meta, after_s=after_s,
                                     force_style="long_fade")
        # The replanned fade must inherit the arm-time flags the old
        # plan carried - grid_fixed keeps the kick-true anchor shift
        # off live-fixed grids, urgent keeps the compressed fade shape.
        for k in ("grid_fixed", "urgent"):
            if k in old:
                plan[k] = old[k]
        plan["gap"] = {"at_rel_s": round(first, 2),
                       "depth_db": late[0][1], "action": "faded"}
    events, swap_at, blend_at = brain.build_events(
        plan, snapshot, active, incoming, cur, cand)
    return plan, events, swap_at, blend_at, plan["gap"]["action"]
