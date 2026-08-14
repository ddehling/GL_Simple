"""What each screen MEASURED on this seam, and what it was measured
against.

`plan_transition` records only a reason string per killed style, which is
enough to count gates but not enough to judge one: "band_clash_high" does
not say whether the seam missed the bar by 0.02 or by 0.9. Rating a
threshold by ear needs the numbers next to the bar.

Every bar here is IMPORTED from where the gate reads it (brain's
KICK_SCREEN_*/BAND_CLASH_*, beatpower's BLEND_MIN*), never retyped - a
panel that quietly disagreed with the engine would be worse than no panel.
The region helpers mirror brain's `_reg_for` for the same reason.

Read-only: nothing here influences a plan.
"""
from lib.dj import beatpower as _bp
from lib.dj.brain import (BAND_CLASH_HI, BAND_CLASH_LO, KICK_SCREEN_BLEND_S,
                          KICK_SCREEN_CUT_S, _CUT_DROP_MAX_BEFORE,
                          _CUT_DROP_MIN_AFTER, _CUT_DROP_MIN_STEP,
                          _kick_delta_s, drop_kick_levels, drop_levels,
                          drop_step)

# Which styles each screen takes off the menu (mirrors brain's kill lists).
_OVERLAP = ("long_blend", "bass_swap", "filter_sweep", "stem_bass_swap",
            "melody_carry", "breakdown_swap", "stem_drum_swap",
            "drum_bridge")
# The beat-power bars lost the plain blends to Gate Check verdicts
# (A-side 2026-08-07, B-side 2026-08-13); they still govern the stem +
# mid-running styles, which is what a trial of them can pin now.
_STEM_OVERLAP = tuple(s for s in _OVERLAP
                      if s not in ("long_blend", "bass_swap",
                                   "filter_sweep"))
_SHORT_DUAL = ("cut_at_drop", "echo_out", "loop_build", "loop_roll_exit",
               "loop_in")

# Bands each style genuinely runs together (mirrors brain's _style_bands).
_STYLE_BANDS = {
    "long_blend": ("high",), "bass_swap": ("high",),
    "filter_sweep": ("high",), "stem_bass_swap": ("mid", "high"),
    "melody_carry": ("mid", "high"), "breakdown_swap": ("mid", "high"),
    "stem_drum_swap": ("low", "mid", "high"),
    "drum_bridge": ("low", "mid", "high"),
}


def region_for(track, at_s, kind):
    """brain._reg_for: the analysed region a seam anchor falls in."""
    try:
        pts = track.mix_outs if kind == "out" else track.mix_ins
        ref = pts[0]["time_s"] if pts else None
    except Exception:
        ref = None
    return kind if (ref is not None and abs(at_s - ref) <= 45.0) else "mid"


def local_phase_known(a, b, out_s, in_s):
    """brain's `_local_ok` - the kick screens stand down where this holds,
    because the kick-true anchors already correct the placement."""
    return (_bp.phase_offset(a.id, at_s=out_s) is not None
            and _bp.phase_offset(b.id, at_s=in_s) is not None)


def _row(name, fired, detail, bar, kills, testable=True, skipped=None):
    return {"gate": name, "fired": bool(fired), "detail": detail,
            "bar": bar, "kills": kills, "testable": testable,
            "skipped": skipped}


def probe(a, b, plan):
    """[{gate, fired, detail, bar, kills, testable, skipped}] for the
    screens a Gate Check session can put on trial. `plan` supplies out_s /
    in_s / rate, so the numbers match the seam actually rendered."""
    # The gates ran on the PRE-SNAP pair anchors; plan["out_s"]/["in_s"]
    # have since moved by the phrase snap and the kick-true offset, and
    # phase_offset() is time-bucketed, so re-checking at the final anchors
    # disagrees with what actually happened on ~6% of seams. diag carries
    # the originals for exactly this.
    _d = plan.get("diag") or {}
    out_s = _d.get("pair_out_s", plan.get("out_s", 0.0))
    in_s = _d.get("pair_in_s", plan.get("in_s", 0.0))
    rate = plan.get("rate", 1.0) or 1.0
    reg_a = region_for(a, out_s, "out")
    reg_b = region_for(b, in_s, "in")
    rows = []

    # -- the kick screens ------------------------------------------------
    local_ok = local_phase_known(a, b, out_s, in_s)
    d = _kick_delta_s(a, b, rate)
    raw = abs((a.kick_offset_s or 0.0) - (b.kick_offset_s or 0.0))
    note = (f"beat-phase distance {1000*d:.1f}ms  "
            f"(A {1000*(a.kick_offset_s or 0):.1f}ms vs B "
            f"{1000*(b.kick_offset_s or 0):.1f}ms/{rate:.4f}, wrapped mod "
            f"A's {1000*(getattr(a, 'period_s', 0) or 0):.0f}ms beat; "
            f"raw linear pre-stretch {1000*raw:.1f}ms)")
    for nm, bar_v, kills in (("kick_offset>20ms", KICK_SCREEN_BLEND_S,
                              _OVERLAP),
                             ("kick_offset>28ms", KICK_SCREEN_CUT_S,
                              _SHORT_DUAL)):
        rows.append(_row(
            nm, (d > bar_v) and not local_ok, note,
            f"≤ {1000*bar_v:.0f}ms", kills,
            skipped=("local phase measured — kick-true anchors already "
                     "correct this" if local_ok else None)))

    # -- beat power, per side, AT THE SEAM'S OWN POSITION ------------------
    # Mirrors brain 2026-08-12: the dense phase-corrected profile, read
    # +/-10s off the boundary (the legacy window centers), stands ALONE
    # when present; the labeled-region max() is only the fallback for
    # unscanned tracks.
    for t, side, reg, _at, bar_v in (
            (a, "A", reg_a, out_s - 10.0, _bp.BLEND_MIN_EXIT),
            (b, "B", reg_b, in_s + 10.0, _bp.BLEND_MIN)):
        pw = _bp.power_at(t.id, _at)
        if pw is not None:
            best, evid, src = pw, [pw], f"@{_at:.0f}s (dense profile)"
        else:
            bs = _bp.band_scores(t.id, region=reg) or {}
            evid = [v for v in (bs.get("low"), _bp.scores().get(t.id))
                    if v is not None]
            best = max(evid) if evid else None
            src = f"{reg}-region (no dense profile)"
        rows.append(_row(
            f"no_beat_power_{side}", bool(evid) and best < bar_v,
            (f"{t.title[:30]} low-band beat power {best:.2f} {src}"
             if best is not None else "not measured"),
            f"≥ {bar_v:.2f}", _STEM_OVERLAP,
            skipped=None if evid else "no beat-power measurement on file"))

    # -- band clash, per band --------------------------------------------
    ba = _bp.band_scores(a.id, region=reg_a) or {}
    bb = _bp.band_scores(b.id, region=reg_b) or {}
    for band in ("low", "mid", "high"):
        va, vb = ba.get(band), bb.get(band)
        kills = tuple(s for s, bands in _STYLE_BANDS.items()
                      if band in bands)
        if va is None or vb is None:
            rows.append(_row(f"band_clash_{band}", False,
                             "not measured", "—", kills,
                             skipped="band scores missing"))
            continue
        hi_, lo_ = max(va, vb), min(va, vb)
        who = "A" if va >= vb else "B"
        rows.append(_row(
            f"band_clash_{band}", hi_ >= BAND_CLASH_HI and lo_ < BAND_CLASH_LO,
            (f"{band}-band rhythmicity  A {va:.2f} / B {vb:.2f}   "
             f"(loud side {who}: {hi_:.2f}, quiet side {lo_:.2f})"),
            f"clash when hi ≥ {BAND_CLASH_HI} and lo < {BAND_CLASH_LO}",
            kills))

    # (cut_needs_grid_conf>=0.8 was probed here from 2026-08-12 until the
    # night of 2026-08-13, when 14 trials rated it right 3 times and the
    # gate was removed - cut_at_drop now sits on the tier's grid_conf<0.7.
    # The row is gone rather than left reading 'never fires': a probe that
    # reports a gate the engine no longer has is worse than no probe.)

    # (stretch>5.5%_risky was probed here 2026-08-12 and removed from the
    # engine on the night of 2026-08-13 after 13 trials rated it right zero
    # times. Its row goes with it - a probe reporting a gate the engine no
    # longer has is worse than no probe. What still bounds stretch is the
    # deck's own [0.90, 1.10] clip and rate_for's 6% per-side cap, neither
    # of which is a rateable threshold.)

    # -- cut_at_drop entry shape (2026-08-14) -----------------------------
    # Only meaningful when the seam IS a cut: the row judges B's entry as
    # a drop, and on any other style in_s is not claiming to be one. The
    # anchor is plan["in_s"] itself, NOT the pre-snap pair anchor - the
    # cut enters exactly at the vetted (or trial near-miss) downbeat.
    if plan.get("style") == "cut_at_drop":
        db = plan.get("in_s", 0.0)
        step = drop_step(b, db)
        lv = drop_levels(b, db)
        pb, pa = drop_kick_levels(b.id, db)
        if step is None or lv is None:
            rows.append(_row("cut_drop_shape", False, "no energy curve",
                             "—", ("cut_at_drop",),
                             skipped="entry shape unmeasurable"))
        else:
            fails = []
            if step < _CUT_DROP_MIN_STEP:
                fails.append("step")
            if lv[1] < _CUT_DROP_MIN_AFTER:
                fails.append("after")
            if lv[0] > _CUT_DROP_MAX_BEFORE:
                fails.append("before")
            # Kick dip->landing is MEASUREMENT ONLY: its kill was rated
            # away 2026-08-14 (fine at x0.06, bad at x0.95 - sorts
            # nothing). It stays in the detail so every future verdict
            # keeps the number beside it.
            kick = (f"kick {pb:.2f}->{pa:.2f} (x{pa / max(pb, 1e-3):.2f})"
                    if pb is not None and pa is not None
                    else "kick not measured")
            rows.append(_row(
                "cut_drop_shape", bool(fails),
                (f"entry @{db:.0f}s  step x{step:.2f}  "
                 f"run-up {lv[0]:.2f} -> lands {lv[1]:.2f} of p95  {kick}"
                 + (f"   FAILS: {', '.join(fails)}" if fails else "")),
                (f"step ≥ {_CUT_DROP_MIN_STEP}, land ≥ "
                 f"{_CUT_DROP_MIN_AFTER}, run-up ≤ {_CUT_DROP_MAX_BEFORE}"),
                ("cut_at_drop",)))
    return rows


def gate_styles(gate):
    """Styles a given screen takes off the menu - i.e. what a trial of it
    can pin. Mirrors the kill lists used in probe(), so a trial cannot pin
    a style the gate never touched (which would rate nothing)."""
    if gate.startswith("kick_offset>28"):
        return _SHORT_DUAL
    if gate.startswith("kick_offset>20"):
        return _OVERLAP
    if gate.startswith("no_beat_power"):
        return _STEM_OVERLAP
    if gate.startswith("band_clash_"):
        band = gate.rsplit("_", 1)[-1]
        return tuple(s for s, bands in _STYLE_BANDS.items() if band in bands)
    if gate.startswith("cut_drop_shape"):
        return ("cut_at_drop",)
    return ()


def gate_names():
    """Screens a Gate Check session can put on trial, most-costly first
    (measured share of seams they kill, 2026-08-07; stretch measured
    2026-08-12). cut_drop_shape was rated 2026-08-14 (25 trials, wrong
    20 times): the bars moved to the rated band's floors and the kick
    kill came off. It stays listed because the NEW bars are themselves
    unrated - the trial now serves the next unheard band (step
    1.25-1.5, land 0.40-0.50, run-up 0.65-0.75)."""
    return ["band_clash_high", "no_beat_power_A", "kick_offset>20ms",
            "no_beat_power_B", "band_clash_mid", "band_clash_low",
            "kick_offset>28ms", "cut_drop_shape"]
