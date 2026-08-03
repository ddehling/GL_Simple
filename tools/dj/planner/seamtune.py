"""Learning the EXECUTION of a seam, not just the choice of one.

Two models, and the difference between them matters more than either:

  * KNOB RESPONSE is a randomised experiment. Every lab seam nudges a
    couple of `build_events` knobs off their defaults by a small amount
    chosen at random, independent of the music. Because the nudge cannot
    be caused by the pair, a correlation between nudge direction and
    verdict is CAUSAL evidence about that knob - "the swap wants to sit
    later than 0.5" - with no repeated pairs needed.

  * CONTEXT COEFFICIENTS are observational. A ridge fit over the seam's
    features (key fit, stretch, groove, grid confidence, blend length...)
    says what a good seam tends to look like, but the brain CHOSE those
    conditions, so the arrows may point either way. Useful for reading,
    not for turning into rules.

Neither model references a song or an artist: everything is a feature of
the seam, so what it learns transfers to material never rated.
"""
import math
import random

import numpy as np

from lib.dj import tuning
from lib.dj.brain import TUNE_DEFAULTS


def learn(rows):
    """Fold the randomised evidence into the live baseline. Returns the
    changes made (empty until a knob clears 2 sigma)."""
    return tuning.apply_findings(knob_findings(rows), TUNE_DEFAULTS, RANGES)

# Which knobs actually affect which styles' automation. Jittering a knob
# a style never reads would log a nudge that provably did nothing and
# dilute the evidence for the ones that matter.
# beats_scale only reaches the styles whose geometry is measured in blend
# BEATS. long_fade, echo_out and the cut family use fixed lead times and
# ignore plan["beats"] entirely (measured: 0/186 probes changed), so
# offering it to them would log nudges that provably did nothing.
_FADE = ("fade_recede", "fade_lead_a", "fade_lead_b", "fade_b_stage1",
         "fade_b_ramp1", "fade_b_ramp2", "fade_out_ramp", "fade_stop_lead")
_BLEND = ("swap_pos", "swap_beats_long", "swap_beats", "trim_cap",
          "b_mid0", "b_mid0_hot", "b_mid0_long",
          "b_high0", "b_high0_hot", "b_high0_long",
          "stage1_gain", "stage1_frac", "high_swap_at", "beats_scale",
          "pre_dip_at", "pre_dip_gain", "exit_res", "exit_res_long")
_ECHO = ("echo_lead_beats", "echo_b_gain", "echo_delay_beats",
         "echo_feedback", "echo_wet", "echo_tail_s")
_CUT = ("brake_s", "brake_chance")
_SPIN = ("spinback_s",)
_ROLL = _BLEND + ("roll_shrink1", "roll_shrink2")
# Styles whose plan can carry a vocal duck get the duck knobs too.
_DUCKERS = ("long_blend", "bass_swap", "filter_sweep", "stem_bass_swap",
            "melody_carry", "loop_in", "breakdown_swap")
_BY_STYLE = {"long_fade": _FADE, "echo_out": _ECHO, "phrase_cut": _CUT,
             "spinback_cut": _SPIN, "loop_roll_exit": _ROLL}

# name -> (low, high) sampling range. Deliberately narrow: these are
# SUBTLE variants of the baseline, not a search of the whole space - a
# wild value would just be rated bad for being wild.
RANGES = {
    "swap_pos": (0.36, 0.64),
    "swap_beats_long": (4.0, 8.0),
    "swap_beats": (3.0, 6.0),
    "trim_cap": (1.12, 1.78),
    "b_mid0": (0.30, 0.62),
    "b_mid0_hot": (0.18, 0.45),
    "b_mid0_long": (0.18, 0.45),
    "b_high0": (0.72, 1.00),
    "b_high0_hot": (0.50, 0.90),
    "b_high0_long": (0.32, 0.70),
    "fade_recede": (0.34, 0.68),
    "fade_lead_a": (5.0, 12.0),
    "fade_lead_b": (2.0, 7.0),
    "fade_b_stage1": (0.42, 0.80),
    "fade_b_ramp1": (2.0, 6.0),
    "fade_b_ramp2": (5.0, 12.0),
    "fade_out_ramp": (3.0, 8.0),
    "fade_stop_lead": (4.0, 9.0),
    "stage1_gain": (0.80, 1.00),
    "stage1_frac": (0.22, 0.50),
    "high_swap_at": (0.12, 0.36),
    "beats_scale": (0.62, 1.55),
    "pre_dip_at": (0.34, 0.68),
    "pre_dip_gain": (0.72, 0.98),
    "exit_res": (4.0, 14.0),
    "exit_res_long": (10.0, 24.0),
    "duck_depth": (0.0, 0.42),
    "duck_beats": (1.0, 4.0),
    "echo_lead_beats": (8.0, 18.0),
    "echo_b_gain": (0.75, 1.00),
    "echo_delay_beats": (0.5, 1.0),
    "echo_feedback": (0.42, 0.78),
    "echo_wet": (0.55, 0.95),
    "echo_tail_s": (1.5, 4.0),
    "spinback_s": (0.9, 2.2),
    "brake_s": (0.5, 1.5),
    "brake_chance": (0.0, 1.0),
    "roll_shrink1": (10.0, 22.0),
    "roll_shrink2": (18.0, 30.0),
}

N_PER_SEAM = 2          # knobs nudged at once - few enough to attribute
MIN_KNOB_N = 12         # below this a knob's direction is not reportable


def knobs_for(style, duck=False):
    """Knobs whose value this style's automation actually reads. Jittering
    one a style never looks at would log a nudge that provably did nothing
    and dilute the evidence for the knobs that matter."""
    ks = _BY_STYLE.get(style, _BLEND)
    if duck and style in _DUCKERS:
        ks = ks + ("duck_depth", "duck_beats")
    return ks


def sample_tune(style, rng=None, width=0.5, duck=False):
    """A subtle variant of the CURRENT baseline for one seam.

    Centred on whatever the knob is worth today - the original constant
    at first, the learned value once evidence has moved it - so search
    keeps happening around the moving baseline instead of re-exploring
    the whole range forever. Always clipped to the absolute range."""
    rng = rng or random
    pool = [k for k in knobs_for(style, duck) if k in RANGES]
    if not pool:
        return {}
    base = tuning.current(TUNE_DEFAULTS)
    out = {}
    for k in rng.sample(pool, min(N_PER_SEAM, len(pool))):
        lo, hi = RANGES[k]
        cur = float(base.get(k, TUNE_DEFAULTS[k]))
        half = (hi - lo) * width / 2.0
        out[k] = round(min(hi, max(lo, rng.uniform(cur - half,
                                                   cur + half))), 4)
    return out


def apply_plan_knobs(plan):
    """Apply the jittered knobs that live in the PLAN rather than in
    build_events. `beats_scale` is the only one: plan["beats"] sizes the
    audition pre-roll and the drawn timeline, so it has to be resolved
    here - and because the lab jitters AFTER plan_transition has run, it
    would otherwise be logged as an experiment that changed nothing."""
    scale = (plan.get("tune") or {}).get("beats_scale")
    if scale and plan.get("beats"):
        plan["beats"] = max(8, int(round(plan["beats"] * float(scale)
                                         / 4.0)) * 4)
    return plan


def _score(verdict):
    return {"good": 1.0, "passable": 0.5, "bad": 0.0}.get(verdict)


def knob_findings(rows, min_n=MIN_KNOB_N):
    """Per knob: does nudging it up help or hurt? Randomised, so causal.

    Reported as the correlation between the nudge (signed, normalised to
    the sampling range) and the verdict score, plus what the evidence
    suggests the value should be - the score-weighted mean of the values
    actually tried."""
    per = {}
    for r in rows:
        tune = r.get("tune") or {}
        s = _score(r.get("verdict"))
        if s is None:
            continue
        for k, v in tune.items():
            if k not in RANGES:
                continue
            lo, hi = RANGES[k]
            if hi <= lo:
                continue
            per.setdefault(k, []).append(((float(v) - lo) / (hi - lo), s,
                                          float(v)))
    out = []
    for k, obs in per.items():
        n = len(obs)
        if n < min_n:
            out.append({"knob": k, "n": n, "thin": True,
                        "default": TUNE_DEFAULTS[k]})
            continue
        xs = np.array([o[0] for o in obs])
        ys = np.array([o[1] for o in obs])
        if xs.std() < 1e-9 or ys.std() < 1e-9:
            out.append({"knob": k, "n": n, "thin": True,
                        "default": TUNE_DEFAULTS[k]})
            continue
        r_xy = float(np.corrcoef(xs, ys)[0, 1])
        # Score-weighted centre of the tried values: where the good ones
        # actually sat. Weight above the mean score only, so bad seams
        # pull nothing rather than pulling backwards.
        w = np.clip(ys - ys.mean(), 0.0, None)
        vals = np.array([o[2] for o in obs])
        suggest = float((vals * w).sum() / w.sum()) if w.sum() > 0 else None
        # Standard error of a correlation, for an honest "is this real".
        se = math.sqrt(max(1.0 - r_xy ** 2, 1e-9) / max(n - 2, 1))
        # 2.5 sigma AND a floor on the effect size: a dozen knobs are
        # tested every update, so a plain 2-sigma bar produces a steady
        # trickle of false movers (measured: trim_cap drifted on noise).
        out.append({"knob": k, "n": n, "r": r_xy, "se": se,
                    "solid": abs(r_xy) > 2.5 * se and abs(r_xy) >= 0.12,
                    "default": TUNE_DEFAULTS[k], "suggest": suggest,
                    "mean_score": float(ys.mean()), "thin": False})
    out.sort(key=lambda d: (d.get("thin", False), -abs(d.get("r", 0.0))))
    return out


# Context features - all seam properties, never a song or artist id.
def _features(r):
    st = None if r.get("rate") is None else abs(r["rate"] - 1.0) * 100.0
    rh = r.get("rhythm") or {}
    conf = [r.get("conf_a"), r.get("conf_b")]
    conf = min([c for c in conf if c is not None], default=None)
    return {
        "key fit": r.get("key_fit"),
        "stretch %": st,
        "pitch shift": (None if r.get("pitch_st") is None
                        else abs(r.get("pitch_st") or 0)),
        "pair score": r.get("pair_score"),
        "groove fit": rh.get("score"),
        "flam ms": rh.get("flam_ms"),
        "kick agreement": rh.get("kick_agreement"),
        "grid confidence": conf,
        "blend beats": r.get("beats"),
        "stems on deck": (None if r.get("stems_a") is None
                          else int(bool(r.get("stems_a")))
                          + int(bool(r.get("stems_b")))),
    }


def feature_model(rows, alpha=2.0, min_n=25):
    """Ridge over standardised seam features -> verdict score.

    Observational: the brain chose these conditions, so a coefficient is
    an association, not a lever. Rows missing a feature are filled with
    the column mean (which contributes nothing to that coefficient)."""
    data = [(_features(r), _score(r.get("verdict"))) for r in rows]
    data = [(f, s) for f, s in data if s is not None]
    if len(data) < min_n:
        return {"n": len(data), "need": min_n, "coefs": []}
    names = list(data[0][0])
    cols = {nm: [f.get(nm) for f, _s in data] for nm in names}
    keep = [nm for nm in names
            if sum(1 for v in cols[nm] if v is not None) >= 0.6 * len(data)]
    if not keep:
        return {"n": len(data), "need": min_n, "coefs": []}
    X, used = [], []
    for nm in keep:
        vals = [v for v in cols[nm] if v is not None]
        mu = float(np.mean(vals))
        col = np.array([mu if v is None else float(v) for v in cols[nm]])
        sd = float(col.std())
        if sd < 1e-9:
            continue
        X.append((col - col.mean()) / sd)
        used.append(nm)
    if not used:
        return {"n": len(data), "need": min_n, "coefs": []}
    X = np.vstack(X).T
    y = np.array([s for _f, s in data])
    y_c = y - y.mean()
    A = X.T @ X + alpha * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X.T @ y_c)
    pred = X @ beta
    ss_res = float(((y_c - pred) ** 2).sum())
    ss_tot = float((y_c ** 2).sum())
    coefs = sorted(({"name": nm, "coef": float(b)}
                    for nm, b in zip(used, beta)),
                   key=lambda d: -abs(d["coef"]))
    return {"n": len(data), "coefs": coefs, "baseline": float(y.mean()),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0}


# ------------------------------------------------------------------ html ---
GOOD_C, BAD_C, DIM_C = "#2e8b57", "#c0392b", "#8a8f98"


def report_html(rows):
    h = ["<h4 style='margin:8px 0 2px'>Execution knobs "
         "<span style='font-weight:normal;color:%s'>— randomised, so these "
         "are causal</span></h4>" % DIM_C]
    ks = knob_findings(rows)
    tried = [k for k in ks if not k.get("thin")]
    thin = [k for k in ks if k.get("thin")]
    if tried:
        h.append("<table width='100%' cellspacing='0' cellpadding='2'>"
                 f"<tr style='color:{DIM_C}'><td width='22%'><i>knob</i></td>"
                 f"<td width='10%' align='right'><i>seams</i></td>"
                 f"<td width='22%'><i>direction</i></td>"
                 f"<td><i>reading</i></td></tr>")
        for k in tried:
            up = k["r"] > 0
            arrow = ("higher rates better" if up else "lower rates better")
            col = GOOD_C if k["solid"] else DIM_C
            sug = ("" if k["suggest"] is None else
                   f"tried values that scored best centre on "
                   f"<b>{k['suggest']:.2f}</b> "
                   f"<span style='color:{DIM_C}'>(default "
                   f"{k['default']:.2f})</span>")
            if not k["solid"]:
                sug = (f"<span style='color:{DIM_C}'>not yet distinguishable "
                       f"from noise</span>")
            h.append(f"<tr><td>{k['knob']}</td>"
                     f"<td align='right' style='color:{DIM_C}'>{k['n']}</td>"
                     f"<td style='color:{col}'>{arrow} "
                     f"(r={k['r']:+.2f})</td><td>{sug}</td></tr>")
        h.append("</table>")
    if thin:
        h.append(f"<p style='color:{DIM_C}'>Still collecting: " + " · ".join(
            f"{k['knob']} {k['n']}/{MIN_KNOB_N}" for k in thin) + "</p>")
    if not ks:
        h.append(f"<p style='color:{DIM_C}'>No jittered seams rated yet. "
                 f"Each lab seam nudges {N_PER_SEAM} execution knobs off "
                 f"their defaults at random; once ~{MIN_KNOB_N} seams "
                 f"carry a given knob, its direction becomes readable — "
                 f"and because the nudge is independent of the music, it "
                 f"is evidence about the knob rather than about the "
                 f"pair.</p>")

    # What the ratings have actually CHANGED - the loop closing.
    base = tuning.current(TUNE_DEFAULTS)
    moved = {k: v for k, v in base.items()
             if abs(v - TUNE_DEFAULTS[k]) > 1e-6}
    if moved:
        h.append("<p><b>Baseline the engine now mixes with</b> "
                 f"<span style='color:{DIM_C}'>(learned, damped "
                 f"{int(tuning.STEP * 100)}% per update; live and lab both "
                 f"read it)</span><br>" + " · ".join(
                     f"<b>{k}</b> {TUNE_DEFAULTS[k]:g} → "
                     f"<span style='color:{GOOD_C}'>{v:g}</span>"
                     for k, v in sorted(moved.items())) + "</p>")
    else:
        h.append(f"<p style='color:{DIM_C}'>Baseline is still every "
                 f"original constant — no knob has cleared 2 sigma yet. "
                 f"When one does it moves {int(tuning.STEP * 100)}% of the "
                 f"way toward the evidence, and both the lab and the live "
                 f"engine start mixing with the new value.</p>")

    fm = feature_model(rows)
    h.append("<h4 style='margin:8px 0 2px'>What a good seam looks like "
             f"<span style='font-weight:normal;color:{DIM_C}'>— "
             f"observational, associations not levers</span></h4>")
    if not fm["coefs"]:
        h.append(f"<p style='color:{DIM_C}'>Needs {fm.get('need', 25)} "
                 f"decided verdicts, have {fm['n']}.</p>")
    else:
        h.append(f"<p style='color:{DIM_C}'>Ridge fit over "
                 f"{fm['n']} rated seams, standardised — a coefficient is "
                 f"the verdict shift per 1 SD of that feature. No song or "
                 f"artist is an input, so this transfers to material you "
                 f"have never rated. R²={fm['r2']:.2f}.</p>"
                 "<table width='100%' cellspacing='0' cellpadding='2'>")
        for c in fm["coefs"]:
            col = GOOD_C if c["coef"] > 0 else BAD_C
            bar = "▪" * max(1, min(int(abs(c["coef"]) * 40), 18))
            h.append(f"<tr><td width='26%'>{c['name']}</td>"
                     f"<td width='14%' align='right' style='color:{col}'>"
                     f"{c['coef']:+.3f}</td>"
                     f"<td style='color:{col}'>{bar}</td></tr>")
        h.append("</table>")
    return "".join(h)
