"""Switchable variants of the REAL exit selection, for comparing what the
DJ does now against what it used to do.

APPLIED 2026-08-07: `soft` and `clamp` are now the LIVE behaviour
(brain.BUDGET_TAU_S and system.EXIT_MAX_FRAC), so "current" here means
today's engine and "legacy" reconstructs the old one. `win` remains
unapplied - it is a real defect whose fix measured no improvement, kept
here so that can be re-checked rather than taken on faith.

What was measured, on a 933-track library: seams exited at a median 0.90
of the track and 19% landed past the track's last groove section - while
`best_pair` on its own picks 0.71 with 3.8% past-groove. The gap was the
play-time budget:

  * `Theme.min_play_s`/`max_play_s` are ABSOLUTE seconds with no reference
    to how long the record is. The groove theme's valley draw has a median
    of 330s against a median track of 332s, so `after_s` routinely demands
    ~99% of the song.
  * `best_pair` treated `after_s` as a HARD filter, so those candidates
    were deleted rather than penalized, and when none survived (17-35% of
    seams) `plan_transition` fell back to `duration - 35s` - the comedown
    by construction.

The levers, and where each one now lives:

  soft   after_s is a score penalty, not a filter.
         LIVE, as brain.BUDGET_TAU_S.
  clamp  the drawn budget is capped at a fraction of the track.
         LIVE, as system.EXIT_MAX_FRAC (applied in _draw_exit).
  win    the "still alive?" energy factor would measure the blend window A
         actually plays, instead of the section AFTER out_s (which
         correlates -0.06 with the audio that plays, and 0.89 with the
         audio that follows). A real defect - but on its own it moved
         nothing measurable, so it is NOT applied.

`legacy` and `win` are built by rewriting `Brain.best_pair`'s own source
at import time, with each anchor asserted to match exactly once. If
brain.py changes under this module it raises instead of silently
comparing against stale logic. Nothing here affects a live night: the
rewrites are only ever bound to local functions.
"""
import inspect
import textwrap

import numpy as np

BLEND_S = 45.0          # 64 beats at ~85 bpm: the workhorse blend span.
                        # A window proxy is needed because best_pair runs
                        # BEFORE plan_transition chooses `beats`.
BUDGET_TAU = 60.0       # seconds of earliness the soft penalty tolerates
CLAMP_FRAC = 0.72       # ceiling the clamp puts on the drawn budget


def blend_rel(track, out_s):
    """Mean 2 Hz energy over [out_s - BLEND_S, out_s] - the audio A really
    plays through the blend - as a ratio of the track's own body level.
    None when the curve is missing or too short, so callers fall back to
    today's section-energy rule rather than guessing."""
    c = getattr(track, "_ec_arr", None)
    if c is None:
        c = np.asarray(track.row.get("energy_curve") or [], dtype=np.float64)
        track._ec_arr = c
    if len(c) < 40:
        return None
    ref = c[int(len(c) * 0.25):int(len(c) * 0.75)]
    if not len(ref) or ref.mean() <= 1e-6:
        return None
    i0 = max(int((out_s - BLEND_S) * 2), 0)
    i1 = min(int(out_s * 2), len(c))
    if i1 - i0 < 4:
        return None
    return float(c[i0:i1].mean() / ref.mean())


def budget_pen(out_s, after_s):
    """Soft replacement for the hard after_s filter: exits before the drawn
    budget stay on the table, decaying with how early they are."""
    if after_s is None or out_s >= after_s:
        return 1.0
    return float(np.exp(-(after_s - out_s) / BUDGET_TAU))


_SUBS = {
    # Restore the pre-2026-08-07 hard filter. With it back in place every
    # surviving candidate is >= after_s, so brain's `bud` term is 1.0
    # throughout and no second substitution is needed to neutralise it.
    # RE-ANCHORED 2026-08-18. The old anchor predated the exit-retry's
    # `exclude_out_s` filter, which landed between these two lines and
    # silently took the Exit Compare tab's BASELINE offline - every
    # 'legacy' and 'clamp_only' build raised instead of comparing. The
    # guard did its job (it refused rather than comparing against stale
    # logic); nobody re-anchored it. The substitution now keeps the
    # exclude filter and only restores the hard after_s cut, which is
    # the one thing 'legacy' is meant to model.
    "legacy": [(
        """        outs = list(cur.mix_outs)
        if exclude_out_s is not None:
            outs = [o for o in outs
                    if abs(o["time_s"] - exclude_out_s) > 2.0]
        if not outs:
            return None""",
        """        outs = [o for o in cur.mix_outs
                if after_s is None or o["time_s"] >= after_s]
        if exclude_out_s is not None:
            outs = [o for o in outs
                    if abs(o["time_s"] - exclude_out_s) > 2.0]
        if not outs:
            return None"""
    )],
    # The 2026-08-17 exit-life damp, OFF. This is the isolation for the
    # fade-crater fix: everything else in selection is identical, so any
    # seam where `nolife` and `current` disagree is a seam the damp
    # MOVED - and that set is the entire audible surface of the change.
    # Rating those is focused testing; rating random seams is not.
    "nolife": [(
        """            xlife = EXIT_LIFE_FLOOR + (1.0 - EXIT_LIFE_FLOOR) \\
                * self._exit_life(cur, o["time_s"])""",
        """            xlife = 1.0"""
    )],
    # `ea` stays defined: rhythm_fit uses it further down to match A's and
    # B's section energies, which is a different job from the alive test.
    "win": [(
        """            ea = sec_a.get("energy") or 0.0
            if _body_e > 0.2:
                of *= 0.25 + 0.75 * min(ea / _body_e, 1.0)""",
        """            ea = sec_a.get("energy") or 0.0
            _r = blend_rel(cur, o["time_s"])
            if _r is not None:
                of *= 0.25 + 0.75 * min(_r, 1.0)
            elif _body_e > 0.2:
                of *= 0.25 + 0.75 * min(ea / _body_e, 1.0)"""
    )],
}

# name -> (best_pair rewrites, clamp the drawn budget to the track?)
#
# `clamp` models system.py's EXIT_MAX_FRAC cap, which lives in _draw_exit
# rather than in best_pair - callers here compute after_s themselves, so
# they have to apply it explicitly to match the live path.
VARIANTS = {
    # what ships TODAY: soft budget in best_pair + the clamped draw
    "current":    ((), True),
    # what shipped BEFORE 2026-08-07 - the comparison baseline
    "legacy":     (("legacy",), False),
    # isolations, for attributing a difference to one half of the change
    "soft_only":  ((), False),
    "clamp_only": (("legacy",), True),
    # live PLUS the still-unapplied energy-window fix
    "win":        (("win",), True),
    # the fade-crater damp turned OFF - pair with "current" to see
    # exactly which seams the 2026-08-17 exit-life fix moved
    "nolife":     (("nolife",), True),
}
BASELINE = "legacy"         # the red line in the Exit Compare tab
PROPOSED = "current"        # the green line's default

_cache = {}


def build_best_pair(flags):
    """Compile a best_pair with `flags` applied, from brain.py's own source."""
    from lib.dj.brain import Brain
    key = tuple(flags)
    if key in _cache:
        return _cache[key]
    if not key:
        _cache[key] = Brain.best_pair
        return Brain.best_pair
    # Substitutions run against the RAW source, so the anchors carry
    # brain.py's own indentation verbatim; dedent only afterwards.
    src = inspect.getsource(Brain.best_pair)
    for f in flags:
        for old, new in _SUBS[f]:
            n = src.count(old)
            if n != 1:
                raise RuntimeError(
                    f"exitvariants: anchor for '{f}' matched {n}x in "
                    f"Brain.best_pair - brain.py changed under this module. "
                    f"Re-anchor _SUBS before trusting any result.\n"
                    f"--- expected ---\n{old}")
            src = src.replace(old, new)
    g = dict(Brain.best_pair.__globals__)
    g.update({"blend_rel": blend_rel, "budget_pen": budget_pen,
              "BLEND_S": BLEND_S, "BUDGET_TAU": BUDGET_TAU, "np": np})
    ns = {}
    exec(compile(textwrap.dedent(src), "<exitvariant>", "exec"), g, ns)
    _cache[key] = ns["best_pair"]
    return ns["best_pair"]


def fallback_out(cur, after_s):
    """plan_transition's last resort when best_pair returns None."""
    out_fb = max(cur.duration_s - 35.0, cur.duration_s * 0.6)
    if after_s is not None:
        out_fb = min(max(out_fb, after_s), max(cur.duration_s - 8.0, out_fb))
    return out_fb


def choose_exit(brain, cur, cand, after_s, variant="current"):
    """Where would `variant` exit A? Returns a dict with the exit, the
    effective budget floor, and whether the fallback had to be used."""
    flags, clamp = VARIANTS[variant]
    aft = min(after_s, cur.duration_s * CLAMP_FRAC) if clamp else after_s
    pair = build_best_pair(flags)(brain, cur, cand, after_s=aft)
    if pair is None:
        return {"out_s": fallback_out(cur, aft), "in_s": None,
                "after_s": aft, "fallback": True, "pair": None}
    return {"out_s": pair["out_s"], "in_s": pair["in_s"], "after_s": aft,
            "fallback": False, "pair": pair}


def last_groove_end(track):
    """End of the last section that is real body material - exits past this
    are the ones that read as noodling in the comedown."""
    e = None
    for s in track.sections or []:
        if s.get("kind") in ("groove", "build"):
            e = s["end_s"]
    return e
