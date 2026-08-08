"""Does the kick screen predict the flam it claims to predict?

Answer, as of 2026-08-07: NO - and that is the point of keeping this
around.

The screens at brain.py refuse overlapped-drum styles when two tracks'
kicks sit differently against their own grids. This renders the seams
they refused and measures the actual kick-to-kick distance through the
real stretch, rate ramps and PLL trims (the seamverify instrument -
beats projected into render time through each deck's own position
trace, never onset detection).

What it found across two samples: `kick_offset_s` has NO relationship to
measured flam (|r| < 0.05 both ways, while measured k2k stays 1-11ms and
the predictor reads 12-90ms). It cannot: measure_kick_offset folds BASS
PLACEMENT onto beat phase - median 0.35 beats on this library - which is
a different quantity from kick-vs-grid skew. The ear agreed
independently: Gate Check rated 16 seams the 20ms screen refused, 13
sounded fine, and the three bad ones had the SMALLEST deltas.

So the screen no longer governs long_blend / bass_swap / filter_sweep.
The arithmetic fixes it still carries (rate-scaling, then circular
wrapping over A's beat period) stand on CORRECTNESS - they compute the
quantity the code means to compute, and d_off_p still reads it for
blend-length halving - not on predictive power, which this test says
neither form has.

This test therefore reports the correlation and refuses to rank two
nulls. It fails only if the quantity turns out to predict flam and the
LIVE form tracks it worse than the raw one.

    python tools/tests/_dj_kickdelta_test.py --music D:/Devel/music -n 24

Population: the screens carry `and not _local_ok`, so they only act
where local phase is unmeasured - phase-profiled pairs are excluded or
this would measure the kick-true anchors instead.
"""
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from lib.dj import beatpower as bp                      # noqa: E402
from lib.dj.brain import Brain, load_library, _kick_delta_s   # noqa: E402
from lib.dj.db import LibraryDB                         # noqa: E402
from lib.dj.themes import get_theme                     # noqa: E402

RATE = 44100
BLEND_STYLES = ("long_blend", "bass_swap", "filter_sweep")
THROTTLE_S = 1.5          # the sweep must coexist with a desktop in use


def raw_delta(a, b):
    """What the screens compared before 2026-08-07."""
    return abs((a.kick_offset_s or 0.0) - (b.kick_offset_s or 0.0))


def local_phase_known(a, b, out_s, in_s):
    """`_local_ok` - the screens stand down where this is True."""
    return (bp.phase_offset(a.id, at_s=out_s) is not None
            and bp.phase_offset(b.id, at_s=in_s) is not None)


def measured_k2k(a, b, plan, decks, marks):
    """Median kick-to-kick distance across the overlap, in ms, measured on
    the rendered audio through each deck's real position trace. None when
    the overlap carries too few beats to judge."""
    from lib.dj.seamverify import _beats_in_span, _render_times
    blend_s, swap_s = marks["blend_s"], marks["swap_s"]
    ticks = {}
    for deck, track, anchor in (("a", a, plan["out_s"]),
                                ("b", b, plan["in_s"])):
        trace = marks["pos"][deck]
        if len(trace) < 2:
            return None
        off = bp.phase_offset(track.id, at_s=anchor) or 0.0
        s_lo = min(p[1] for p in trace)
        s_hi = max(p[1] for p in trace)
        beats = _beats_in_span(track, s_lo, s_hi)
        if not len(beats):
            return None
        ticks[deck] = _render_times(beats + off, trace)
    ka, kb = ticks["a"], ticks["b"]
    sa = ka[(ka >= blend_s) & (ka <= swap_s)]
    sb = kb[(kb >= blend_s) & (kb <= swap_s)]
    if len(sa) < 4 or len(sb) < 4:
        return None
    # Distance to the NEAREST B beat: a flam is a near-miss, so the
    # relevant quantity is |offset| from the closest partner, not from a
    # nominally-paired index (a tempo-multiple misread would show as a
    # sawtooth, which the median absorbs).
    return float(np.median([np.min(np.abs(sb - x)) for x in sa])) * 1000.0


def build_cases(db, lib, want, seed):
    """Seams where the screens actually ACT (local phase unmeasured), with
    a real stretch, spread across the raw/corrected disagreement."""
    import time
    theme = get_theme("groove")
    rng = random.Random(seed)
    brain = Brain(lib, theme, seed=seed)
    pool = [t for t in lib if 150 <= t.duration_s <= 420
            and t.mix_outs and t.sections]
    cases, tries = [], 0
    while len(cases) < want and tries < 4000:
        tries += 1
        a = rng.choice(pool)
        b, meta = brain.choose_next(a, 0.5, a.bpm)
        if b is None or not (150 <= b.duration_s <= 420):
            continue
        rate = (meta or {}).get("rate", 1.0) or 1.0
        if abs(np.log(rate)) < 0.008:
            continue                     # no stretch - nothing to correct
        # Force a blend style THROUGH the gates: the point is to hear what
        # the screen would have refused, so the screen must not veto the
        # render. test_gates is the sanctioned offline override.
        want_style = rng.choice(BLEND_STYLES)
        try:
            plan = brain.plan_transition(a, b, dict(meta),
                                         after_s=a.duration_s * 0.55,
                                         force_style=want_style,
                                         test_gates=True)
        except Exception:
            continue
        if plan["style"] not in BLEND_STYLES or int(plan.get("beats") or 0) < 16:
            continue
        if local_phase_known(a, b, plan["out_s"], plan["in_s"]):
            continue                     # anchors already fix it - not the
                                         # population the screen governs
        cases.append({"a": a, "b": b, "plan": plan, "rate": rate,
                      "raw": raw_delta(a, b),
                      "corr": _kick_delta_s(a, b, rate)})
    return cases


def main():
    music = "D:/Devel/music"
    if "--music" in sys.argv:
        music = sys.argv[sys.argv.index("--music") + 1]
    want = 16
    if "-n" in sys.argv:
        want = int(sys.argv[sys.argv.index("-n") + 1])
    seed = 7
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])

    db = LibraryDB(music)
    lib = load_library(db)
    print(f"library: {len(lib)} tracks")
    cases = build_cases(db, lib, want, seed)
    print(f"{len(cases)} stretched seams where the kick screen actually acts "
          f"(local phase unmeasured)\n")
    if len(cases) < 4:
        print("FAILED: not enough cases to judge - widen -n or the seed")
        return 1

    import time
    from tools.dj.dj_knobsweep import render_tapped
    rows = []
    for i, c in enumerate(cases, 1):
        a, b, plan = c["a"], c["b"], c["plan"]
        print(f"[{i}/{len(cases)}] {a.title[:30]} -> {b.title[:26]}  "
              f"{plan['style']} rate={c['rate']:.4f}  "
              f"raw={1000*c['raw']:.1f}ms corrected={1000*c['corr']:.1f}ms",
              flush=True)
        try:
            _mix, decks, marks = render_tapped(db, a, b, dict(plan))
            k2k = measured_k2k(a, b, plan, decks, marks)
        except Exception as e:
            print(f"      render failed: {type(e).__name__}: {e}")
            k2k = None
        if k2k is None:
            print("      (no measurable overlap - skipped)")
        else:
            print(f"      MEASURED kick-to-kick {k2k:.1f}ms")
            rows.append((c, k2k))
        time.sleep(THROTTLE_S)

    if len(rows) < 4:
        print("\nFAILED: too few measurable seams")
        return 1

    raw = np.array([1000 * c["raw"] for c, _ in rows])
    cor = np.array([1000 * c["corr"] for c, _ in rows])
    k2k = np.array([k for _, k in rows])
    print("\n" + "=" * 66)
    print(f"{len(rows)} rendered seams")
    print(f"  measured kick-to-kick: median {np.median(k2k):.1f}ms  "
          f"range {k2k.min():.1f}-{k2k.max():.1f}ms")

    def score(name, pred):
        err = float(np.mean(np.abs(pred - k2k)))
        r = (float(np.corrcoef(pred, k2k)[0, 1])
             if np.std(pred) > 1e-9 and np.std(k2k) > 1e-9 else float("nan"))
        print(f"  {name:<26} mean|error| {err:6.2f}ms   corr {r:+.3f}")
        return err, r

    e_raw, r_raw = score("raw stored offsets", raw)
    e_cor, r_cor = score("rate-corrected (live)", cor)

    flips = [(c, k) for c, k in rows
             if (c["raw"] > 0.020) != (c["corr"] > 0.020)]
    print(f"\n  seams whose 20ms verdict FLIPS: {len(flips)}")
    agree = 0
    for c, k in flips:
        new_blocks = c["corr"] > 0.020
        # The screen exists because flam turns audible ~25ms; that is the
        # line the measurement is judged against, not the 20ms screen
        # (which carries deliberate headroom for its own error).
        really_bad = k > 25.0
        ok = (new_blocks == really_bad)
        agree += ok
        print(f"    {c['a'].title[:26]:<27} rate={c['rate']:.3f}  "
              f"raw {1000*c['raw']:5.1f} -> corr {1000*c['corr']:5.1f}ms  "
              f"measured {k:5.1f}ms  new verdict "
              f"{'BLOCK' if new_blocks else 'ALLOW'}  "
              f"{'agrees' if ok else 'DISAGREES'}")
    print()
    passed = True
    # RANKING TWO NULLS IS NOISE. When neither form correlates with the
    # rendered flam, a 1ms gap in mean error says nothing about which
    # arithmetic is right - it says the quantity does not predict flam,
    # which is the standing finding and agrees with the ear verdicts.
    if max(abs(r_raw), abs(r_cor)) < 0.30:
        print(f"  [INFO] neither form predicts the rendered flam "
              f"(|r| {abs(r_raw):.3f} raw, {abs(r_cor):.3f} live) - "
              f"kick_offset_s measures bass placement, not kick skew, so "
              f"this is expected. Nothing to rank.")
    elif e_cor <= e_raw:
        print(f"  [PASS] the live form tracks the measurement at least as "
              f"closely ({e_cor:.2f} vs {e_raw:.2f} ms mean error)")
    else:
        passed = False
        print(f"  [FAIL] raw offsets track the measurement better "
              f"({e_raw:.2f} vs {e_cor:.2f} ms) AND the quantity is "
              f"predictive (r={r_cor:+.3f}) - the correction is wrong")
    if flips:
        print(f"  [{'PASS' if agree >= len(flips) / 2 else 'FAIL'}] flipped "
              f"verdicts agree with measured flam on {agree}/{len(flips)}")
        passed = passed and agree >= len(flips) / 2
    print("\n" + ("ALL PASS" if passed else "FAILED"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
