"""DJ-THEORY CONFORMANCE: does the system plan like real DJs mix?

Reference standards, with sources:
  [ISMIR20] Kim et al., "A Computational Analysis of Real-World DJ Mixes"
  (ISMIR 2020, arXiv:2008.10267; 1,557 mixes from 1001Tracklists):
    - 86.1% of tempo adjustments < 5%, 94.5% < 10%
    - transition lengths cluster at MULTIPLES OF 32 BEATS (phrases)
    - only 2.5% of tracks are key-transposed; 94.3% of those by 1 semitone
  [PRACTICE] community canon (Mixed In Key harmonic-mixing guide, DJ.Studio
  Camelot guide, phrase-mixing guides):
    - harmonic mixing: same / +-1 / relative-mode Camelot codes
    - transitions land on phrase boundaries
    - one bassline at a time (audio-level: _dj_quality_test)
    - never blend two vocal passages over each other

Plan-level conformance across brain-suggested AND user-ordered sets.
Audio-level conformance lives in tools/tests/_dj_quality_test.py.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from lib.dj.brain import camelot_compat, load_library
from lib.dj.db import LibraryDB
from lib.dj.setlist import compile_plan, suggest_set
from lib.dj.themes import get_theme

# The SHOW library - measure on the music the DJ actually plays.
MUSIC = "C:/Users/ddehl/Desktop/Devel/music"
import sys as _sys
if "--music" in _sys.argv:
    MUSIC = _sys.argv[_sys.argv.index("--music") + 1]
BLEND_FAM = ("long_blend", "bass_swap", "filter_sweep", "loop_roll_exit")

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def collect_seams(library, theme):
    """Seams from 6 brain-suggested + 6 user-ordered plans."""
    seams = []
    for seed in range(6):
        entries = suggest_set(library, theme, minutes=35, seed=seed)
        plan = compile_plan(library, entries, theme, seed=seed)
        seams += [("brain", plan["slots"], k)
                  for k in range(len(plan["slots"]) - 1)
                  if plan["slots"][k]["transition"]]
        picks = random.Random(seed + 50).sample(library, 10)
        entries = [{"track_id": t.id, "pin_type": "anchor",
                    "target_offset_min": None, "style_override": None}
                   for t in picks]
        plan = compile_plan(library, entries, theme, seed=seed)
        seams += [("user", plan["slots"], k)
                  for k in range(len(plan["slots"]) - 1)
                  if plan["slots"][k]["transition"]]
    return seams


def main():
    db = LibraryDB(MUSIC)
    library = load_library(db)
    theme = get_theme("groove")
    seams = collect_seams(library, theme)
    matched = [(o, sl, k) for o, sl, k in seams
               if sl[k]["transition"]["style"] != "long_fade"]
    print(f"conformance set: {len(seams)} seams "
          f"({len(matched)} beat-matched) from 12 plans\n")

    # -- [ISMIR20] tempo-adjustment distribution -------------------------------
    # Real DJs' tempo distribution reflects their TRACK SELECTION freedom;
    # gate it where the system selects (brain picks). User-forced orders
    # can't choose their gaps - reported, not gated.
    def _stretch_dist(origin):
        out = []
        for o, sl, k in matched:
            if o != origin:
                continue
            tr = sl[k]["transition"]
            for r in (tr["rate"], tr.get("a_rate", 1.0) or 1.0):
                if abs(r - 1.0) > 1e-4 or r == tr["rate"]:
                    out.append(abs(r - 1.0))
        return out
    br = _stretch_dist("brain")
    us = _stretch_dist("user")
    lt5 = float(np.mean([s < 0.05 for s in br]))
    lt10 = float(np.mean([s < 0.10 for s in br]))
    print(f"  [info] user-ordered stretches (forced by given order): "
          f"{np.mean([s < 0.05 for s in us]) * 100:.0f}% < 5%")
    check("[ISMIR20] tempo adjustments mostly tiny (brain-selected)",
          lt5 >= 0.85 and lt10 >= 0.945,
          f"{lt5 * 100:.1f}% < 5% (real DJs: 86.1%), "
          f"{lt10 * 100:.1f}% < 10% (real: 94.5%)")

    # -- [ISMIR20] transition lengths on 32-beat phrase multiples --------------
    fam = [sl[k]["transition"]["beats"] for o, sl, k in matched
           if sl[k]["transition"]["style"] in BLEND_FAM]
    on32 = float(np.mean([b % 32 == 0 for b in fam])) if fam else 0.0
    check("[ISMIR20] blend lengths are 32-beat phrase multiples",
          on32 >= 0.95,
          f"{on32 * 100:.0f}% of {len(fam)} blends on 32-multiples "
          f"(real mixes cluster at every 32 beats)")

    # -- [ISMIR20] key transposition is rare, and only 1 semitone --------------
    shifted = [abs(sl[k]["transition"].get("pitch_st", 0) or 0)
               for o, sl, k in matched]
    frac_shift = float(np.mean([s > 0 for s in shifted]))
    only1 = all(s <= 1 for s in shifted)
    check("[ISMIR20] pitch transposition rare + 1 semitone max",
          frac_shift <= 0.10 and only1,
          f"{frac_shift * 100:.1f}% of seams shifted (real DJs: 2.5%), "
          f"max {max(shifted) if shifted else 0} st (real: 94.3% are 1 st)")

    # -- [PRACTICE] phrase-boundary alignment ----------------------------------
    aligned = total = 0
    for o, sl, k in matched:
        t = sl[k]["track"]
        tr = sl[k]["transition"]
        # drop-anchored styles exit ON the drop by design - also DJ canon
        if tr["style"] == "loop_build":
            continue
        if t.phrase_beats <= 0 or t.phrase_conf < 0.1:
            continue
        total += 1
        span = t.phrase_beats * t.period_s
        out_s = sl[k]["transition"]["out_s"]
        off = abs(((out_s - t.phrase_start_s + span / 2) % span) - span / 2)
        if off <= 2 * t.period_s + 0.05:   # grid-vs-hypermeter tolerance
            aligned += 1
    check("[PRACTICE] exits land on detected phrase boundaries",
          total == 0 or aligned / total >= 0.9,
          f"{aligned}/{total} phrase-aligned (tracks with hypermeter)")

    # -- [PRACTICE] harmonic mixing on brain-chosen seams -----------------------
    compat = []
    for o, sl, k in matched:
        if o != "brain":
            continue
        tr = sl[k]["transition"]
        from lib.dj.brain import _shift_camelot
        b_cam = _shift_camelot(sl[k + 1]["track"].camelot,
                               tr.get("pitch_st", 0) or 0)
        compat.append(camelot_compat(sl[k]["track"].camelot, b_cam))
    frac_ok = float(np.mean([c >= 0.55 for c in compat]))
    check("[PRACTICE] brain picks harmonically compatible keys",
          frac_ok >= 0.70,
          f"{frac_ok * 100:.0f}% of {len(compat)} brain seams compatible "
          f"(same/adjacent/relative Camelot)")

    # -- [PRACTICE] never blend two vocal passages -----------------------------
    clashes = checked = 0
    for o, sl, k in matched:
        a, b = sl[k]["track"], sl[k + 1]["track"]
        tr = sl[k]["transition"]
        if tr.get("style") == "long_fade":
            # The dipped fade overlaps ~2s at low level - vocal into
            # vocal THROUGH the dip is a handoff, not a fight (it's the
            # planner's designated escape for exactly these pairs).
            continue
        sa = a.section_at(tr["out_s"] - 1.0) or {}
        sb = b.section_at(tr["in_s"] + 1.0) or {}
        if not (a.axes.get("vocal_src") and b.axes.get("vocal_src")):
            continue
        checked += 1
        if (sa.get("vocalness") or 0) > 0.5 and (sb.get("vocalness") or 0) > 0.5:
            clashes += 1
    check("[PRACTICE] no vocal-over-vocal blends",
          clashes == 0,
          f"{clashes}/{checked} seams blend two vocal passages "
          f"(measured vocal data on both sides)")

    # -- [PRACTICE] flavor steering: themes reach DIFFERENT music ---------------
    # A real DJ's "deep hypnotic night" and "hard driving night" are not
    # the same tracklist. Same seed, flavored themes: low overlap, and
    # each set's tracks measurably lean toward its axis targets.
    # RANKED axes, not raw. The raw values compress into the top of their
    # range on any real collection (hardness sat at exactly 1.0 for 81% of
    # a 649-track library), which is why this check used to excuse itself
    # with "axis means saturate on some libraries" and assert only that the
    # lean wasn't INVERTED. axes_rank is the library percentile the brain
    # actually steers on, so the separation is now a real number and the
    # assertion can be a real one.
    def _axis_mean(theme_name, axis, seed=2):
        th = get_theme(theme_name)
        entries = suggest_set(library, th, minutes=45, seed=seed)
        ids = [e["track_id"] for e in entries]
        by_id = {t.id: t for t in library}
        vals = [by_id[i].axes_rank.get(axis) for i in ids
                if i in by_id and by_id[i].axes_rank.get(axis) is not None]
        return set(ids), float(np.mean(vals)) if vals else 0.5
    hyp_ids, hyp_hypnotic = _axis_mean("hypnotic_deep", "hypnotic")
    hard_ids, hard_hardness = _axis_mean("hard_drive", "hardness")
    _, hyp_hardness = _axis_mean("hypnotic_deep", "hardness")
    _, hard_hypnotic = _axis_mean("hard_drive", "hypnotic")
    overlap = len(hyp_ids & hard_ids) / max(min(len(hyp_ids),
                                                len(hard_ids)), 1)
    check("[PRACTICE] flavored themes pick different music",
          overlap <= 0.5 and hyp_hypnotic > hard_hypnotic
          and hard_hardness > hyp_hardness,
          f"track overlap {overlap * 100:.0f}%; hypnotic rank "
          f"{hyp_hypnotic:.2f} vs {hard_hypnotic:.2f}, hardness rank "
          f"{hard_hardness:.2f} vs {hyp_hardness:.2f}")

    # Live flavor override shifts picks the same way mid-set.
    from lib.dj.brain import Brain
    b0 = Brain(library, get_theme("groove"), seed=4)
    b1 = Brain(library, get_theme("groove"), seed=4)
    b1.set_flavor({"prefer_tags": {"hypnotic": 1.0},
                   "axis_targets": {"hypnotic": 0.95}})
    cur = library[0]
    picks0, picks1 = [], []
    for _ in range(8):
        c0, _m = b0.choose_next(cur, 0.6, cur.bpm)
        c1, _m = b1.choose_next(cur, 0.6, cur.bpm)
        if c0 is not None:
            picks0.append(c0.axes.get("hypnotic", 0.5))
            b0.note_played(c0)
        if c1 is not None:
            picks1.append(c1.axes.get("hypnotic", 0.5))
            b1.note_played(c1)
    check("[PRACTICE] live flavor override steers picks mid-set",
          np.mean(picks1) > np.mean(picks0),
          f"hypnotic-lean picks avg {np.mean(picks1):.2f} vs "
          f"baseline {np.mean(picks0):.2f}")

    print()
    if failures:
        print(f"FAILED: {len(failures)}: {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS - planning conforms to measured real-DJ practice")


if __name__ == "__main__":
    main()
