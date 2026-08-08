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
                          KICK_SCREEN_CUT_S, _kick_delta_s)

# Which styles each screen takes off the menu (mirrors brain's kill lists).
_OVERLAP = ("long_blend", "bass_swap", "filter_sweep", "stem_bass_swap",
            "melody_carry", "breakdown_swap", "stem_drum_swap",
            "drum_bridge")
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

    # -- beat power, per side, in the seam's own region -------------------
    for t, side, reg, bar_v in ((a, "A", reg_a, _bp.BLEND_MIN_EXIT),
                                (b, "B", reg_b, _bp.BLEND_MIN)):
        bs = _bp.band_scores(t.id, region=reg) or {}
        evid = [v for v in (bs.get("low"), _bp.scores().get(t.id))
                if v is not None]
        best = max(evid) if evid else None
        rows.append(_row(
            f"no_beat_power_{side}", bool(evid) and best < bar_v,
            (f"{t.title[:30]} {reg}-region low-band beat power "
             f"{best:.2f}" if best is not None else "not measured"),
            f"≥ {bar_v:.2f}", _OVERLAP,
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
    return rows


def gate_names():
    """Screens a Gate Check session can put on trial, most-costly first
    (measured share of seams they kill, 2026-08-07)."""
    return ["band_clash_high", "no_beat_power_A", "kick_offset>20ms",
            "no_beat_power_B", "band_clash_mid", "band_clash_low",
            "kick_offset>28ms"]
