#!/usr/bin/env python
"""Read the DJ's own night logs and tell you what it actually did.

The system self-assesses every seam it plays (`seam_quality` events: worst
grid flam, level holes, resnaps) and stamps the prediction that produced
that seam (`armed` events: the selection term breakdown, the style menu and
what was gated off it). Nothing read any of it - 14 nights and 560 measured
seams sat on disk teaching nobody. This is the reader.

    python tools/dj/dj_review.py                # every night on disk
    python tools/dj/dj_review.py --last         # just the most recent night
    python tools/dj/dj_review.py --since 7      # last 7 days
    python tools/dj/dj_review.py --terms        # + selection-term validation
    python tools/dj/dj_review.py --gates        # + why styles never reached the menu
    python tools/dj/dj_review.py --skips        # + what the operator kept rejecting
    python tools/dj/dj_review.py --all          # all sections

WHAT THE SECTIONS ARE FOR

  Night summary   what played, how long, how it ended.
  Seam report     styles used vs measured verdicts. The headline number is
                  the long_fade share: it is the system admitting it could
                  not beat-match, and its causes are listed underneath.
  Term validation (--terms) each selection term's value on seams that came
                  out CLEAN vs seams that FLAMMED or HOLED, with a
                  point-biserial correlation against the measured error.
                  A term whose clean and rough means are the same is not
                  discriminating - it is costing candidates score for
                  nothing. This is the evidence for retuning a constant;
                  it is NOT an autotuner, and it says so when a term has
                  too few rough samples to mean anything.
  Gate report     (--gates) how often each style was zeroed off the menu
                  and by which gate. A style that is never gated but never
                  played is losing the dice roll (a weights problem); one
                  that is always gated is unreachable (a gate problem).
                  Two very different fixes, previously indistinguishable.
  Skip report     (--skips) tracks the operator skipped, and - the thing
                  worth acting on - tracks that get RE-OFFERED after being
                  skipped, which is the brain not learning.

Prediction-vs-measurement (`predicted_rhythm` on a seam) is checked in the
term section too: the groove predictor is a claim about a seam made before
it played, and this is the only place that claim is ever scored.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "logs")

# A seam is "rough" by the ENGINE's own bar - the same thresholds
# system._assess_seam uses to hand out an auto thumbs-down - so this report
# and the cross-night learning never disagree about what a bad seam was.
FLAM_BEATS = 0.12
HOLE_S = 1.5
# ...but the codebase's own stated audibility threshold for two percussive
# transients is ~25 ms (brain.plan_transition's groove-offset gate), which
# is ~0.055 beats at 128 bpm - less than HALF the verdict bar. The two
# numbers measure slightly different things (grid-phase error vs kick
# placement), so a seam in the band between them is not proven audible;
# what IS certain is that it is invisible to the learning loop. Counted
# separately rather than folded silently into "clean".
AUDIBLE_BEATS = 0.055


# ---------------------------------------------------------------- loading

def load_nights(since_days=None, last_only=False, log_dir=LOG_DIR):
    """[(date_str, [event, ...]), ...] oldest first."""
    paths = sorted(glob.glob(os.path.join(log_dir, "dj_*.jsonl")))
    paths = [p for p in paths
             if os.path.basename(p)[3:-6].isdigit()]     # dj_YYYYMMDD.jsonl
    if last_only:
        paths = paths[-1:]
    elif since_days:
        paths = paths[-int(since_days):]
    nights = []
    for p in paths:
        evs = []
        with open(p, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    evs.append(json.loads(line))
                except ValueError:
                    continue
        if evs:
            nights.append((os.path.basename(p)[3:-6], evs))
    return nights


def pair_seams(events):
    """Join each `armed` plan to the `seam_quality` it produced.

    The engine logs the plan when it arms and the measurement when the seam
    finishes, with the incoming title as the only shared key (`armed.next`
    == `seam_quality.b`). Walk forward in time and match the first unclaimed
    measurement for that title - an armed transition that got ABORTED never
    produces one and correctly drops out."""
    seams = []
    pending = []                       # armed events awaiting their outcome
    for e in events:
        k = e.get("event")
        if k == "armed":
            pending.append(e)
        elif k == "abort":
            if pending:
                pending.pop()          # recalled before it played
        elif k == "seam_quality":
            hit = None
            for i, a in enumerate(pending):
                if a.get("next") == e.get("b"):
                    hit = pending.pop(i)
                    break
            seams.append({"armed": hit or {}, "q": e})
    return seams


def is_rough(q):
    return (float(q.get("max_err_beats") or 0.0) >= FLAM_BEATS
            or float(q.get("hole_s") or 0.0) >= HOLE_S)


def severity(q):
    """A single 0..1 badness number for correlation. Flam in beats and hole
    in seconds are different units; normalize each by the point it becomes
    AUDIBLE (not the verdict bar - correlation wants the full gradient, not
    a step at the threshold) and take the worse."""
    f = float(q.get("max_err_beats") or 0.0) / AUDIBLE_BEATS
    h = float(q.get("hole_s") or 0.0) / 0.35
    return min(max(f, h), 3.0) / 3.0


# ---------------------------------------------------------------- reports

def fmt_pct(n, d):
    return f"{n:4d}  {(100.0 * n / d if d else 0):5.1f}%"


def report_nights(nights):
    print("=" * 72)
    print("NIGHTS")
    print("=" * 72)
    tot_seams = tot_plays = 0
    for date, evs in nights:
        c = Counter(e.get("event") for e in evs)
        clock = max((float(e.get("clock_s") or 0.0) for e in evs), default=0.0)
        themes = [e.get("theme") for e in evs if e.get("event") == "theme"]
        seams = [e for e in evs if e.get("event") == "seam_quality"]
        rough = sum(1 for e in seams if is_rough(e))
        tot_seams += len(seams)
        tot_plays += c.get("play", 0)
        print(f"  {date}  {clock/3600:4.1f}h  "
              f"{c.get('play', 0):3d} tracks  {len(seams):3d} seams  "
              f"{rough:2d} rough  {c.get('skip', 0):3d} skips  "
              f"{c.get('abort', 0):2d} aborts  "
              f"{c.get('flam_bailout', 0):2d} bailouts"
              + (f"  [{'/'.join(dict.fromkeys(t for t in themes if t))}]"
                 if themes else ""))
    print(f"  {'-' * 66}")
    print(f"  TOTAL: {len(nights)} nights, {tot_plays} plays, "
          f"{tot_seams} measured seams")


def report_seams(seams):
    print()
    print("=" * 72)
    print("SEAMS")
    print("=" * 72)
    if not seams:
        print("  no measured seams")
        return
    n = len(seams)
    by_style = defaultdict(list)
    for s in seams:
        by_style[s["q"].get("style") or "?"].append(s)
    print(f"  {'style':18} {'used':>10}   {'rough':>5}   "
          f"{'med flam':>8}  {'worst':>6}")
    for style, group in sorted(by_style.items(),
                               key=lambda kv: -len(kv[1])):
        errs = [float(s["q"].get("max_err_beats") or 0.0) for s in group]
        rough = sum(1 for s in group if is_rough(s["q"]))
        print(f"  {style:18} {fmt_pct(len(group), n)}   "
              f"{rough:3d}   {st.median(errs):8.3f}  {max(errs):6.3f}")
    verdicts = Counter(s["q"].get("verdict") for s in seams)
    print("\n  engine verdicts : "
          + ", ".join(f"{k} {v}" for k, v in verdicts.most_common()))
    rough = sum(1 for s in seams if is_rough(s["q"]))
    print(f"  rough (engine bar, flam>{FLAM_BEATS} beats "
          f"or hole>{HOLE_S}s): {rough} of {n} ({100.0*rough/n:.0f}%)")
    # The band between "audible" and "charged as a train-wreck": these
    # seams were heard, and taught the system nothing.
    grey = [s for s in seams
            if not is_rough(s["q"])
            and float(s["q"].get("max_err_beats") or 0.0) >= AUDIBLE_BEATS]
    if grey:
        print(f"  in the grey band "
              f"(flam {AUDIBLE_BEATS}-{FLAM_BEATS} beats): "
              f"{len(grey)} of {n} ({100.0*len(grey)/n:.0f}%)")
        print("    -> past the codebase's own ~25ms audibility figure but")
        print("       under the verdict bar, so they charge pair memory")
        print("       nothing. If the room hears these, lower the bar.")
    fades = [s for s in seams if s["q"].get("style") == "long_fade"]
    if fades:
        print(f"\n  LONG_FADE share: {100.0*len(fades)/n:.0f}% "
              "(the system declining to beat-match). Causes:")
        why = Counter(s["armed"].get("fade_reason") or "unlogged"
                      for s in fades)
        for reason, k in why.most_common():
            print(f"    {reason:22} {fmt_pct(k, len(fades))}")
        if why.get("unlogged"):
            print("    (unlogged = seam predates fade-reason stamping)")


def report_terms(seams):
    print()
    print("=" * 72)
    print("SELECTION TERM VALIDATION")
    print("=" * 72)
    have = [s for s in seams if s["armed"].get("terms")]
    if not have:
        print("  No armed events carry a term breakdown yet.")
        print("  Term stamping landed 2026-07-24 - play a night and re-run.")
        return
    clean = [s for s in have if not is_rough(s["q"])]
    rough = [s for s in have if is_rough(s["q"])]
    print(f"  {len(have)} seams with terms  "
          f"({len(clean)} clean / {len(rough)} rough)")
    if len(rough) < 8:
        print(f"  NOT ENOUGH ROUGH SEAMS to conclude anything ({len(rough)});")
        print("  means are shown, correlations are not. Want ~15+.")
    print()
    keys = sorted({k for s in have for k in s["armed"]["terms"]})
    print(f"  {'term':10} {'clean':>7} {'rough':>7} {'delta':>7} "
          f"{'r':>7}   verdict")
    rows = []
    for k in keys:
        cv = [s["armed"]["terms"][k] for s in clean
              if k in s["armed"]["terms"]]
        rv = [s["armed"]["terms"][k] for s in rough
              if k in s["armed"]["terms"]]
        if not cv or not rv:
            continue
        cm, rm = st.mean(cv), st.mean(rv)
        r = _corr([s["armed"]["terms"].get(k) for s in have],
                  [severity(s["q"]) for s in have])
        rows.append((abs(cm - rm), k, cm, rm, r,
                     st.pstdev(cv + rv)))
    for _, k, cm, rm, r, sd in sorted(rows, reverse=True):
        # Every term is meant to score GOOD candidates higher, so a term
        # that is doing its job reads higher on the seams that came out
        # clean. The sign of (clean - rough) is the whole verdict; r only
        # corroborates it, and is suppressed entirely on thin samples
        # rather than being quietly folded in.
        show_r = r is not None and len(rough) >= 8
        if sd < 0.01:
            note = "CONSTANT - no steering"
        elif abs(cm - rm) < 0.01:
            note = "no discrimination"
        elif cm > rm:
            note = "discriminates as designed"
            if show_r and r > 0:
                note += " (but r disagrees)"
        else:
            note = "INVERTED - higher term, worse seam"
            if show_r and r < 0:
                note += " (but r disagrees)"
        rs = f"{r:7.2f}" if show_r else "      -"
        print(f"  {k:10} {cm:7.3f} {rm:7.3f} {cm-rm:+7.3f} {rs}   {note}")
    print("\n  clean/rough = mean term value on seams that measured clean vs")
    print("  rough; r = correlation of the term against measured severity")
    print("  (negative = higher term score, better seam = working as meant).")
    _report_rhythm_prediction(seams)


def _report_rhythm_prediction(seams):
    have = [s for s in seams if (s["q"].get("predicted_rhythm") or {})
            .get("score") is not None]
    if not have:
        return
    print()
    print("  GROOVE PREDICTOR (predicted_rhythm vs what the seam measured)")
    clean = [s for s in have if not is_rough(s["q"])]
    rough = [s for s in have if is_rough(s["q"])]
    for label, group in (("clean", clean), ("rough", rough)):
        if not group:
            continue
        vals = [s["q"]["predicted_rhythm"]["score"] for s in group]
        print(f"    predicted score on {label:5} seams: "
              f"n={len(vals):3d}  mean={st.mean(vals):.3f}")
    r = _corr([s["q"]["predicted_rhythm"]["score"] for s in have],
              [severity(s["q"]) for s in have])
    if r is not None:
        print(f"    correlation with measured severity: r = {r:+.2f}")
        if len(rough) < 8:
            print("    (too few rough seams to trust - keep collecting)")
        elif r > -0.15:
            print("    NOT PREDICTIVE. The groove terms gate styles and lean")
            print("    selection on the strength of this claim; it is not")
            print("    currently earning that authority.")


def _corr(xs, ys):
    """Pearson r over the pairs where both are present."""
    pts = [(x, y) for x, y in zip(xs, ys)
           if isinstance(x, (int, float)) and isinstance(y, (int, float))
           and math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 6:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    sx, sy = st.pstdev(xs), st.pstdev(ys)
    if sx < 1e-9 or sy < 1e-9:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    return (sum((x - mx) * (y - my) for x, y in pts) / len(pts)) / (sx * sy)


def report_gates(seams, events):
    print()
    print("=" * 72)
    print("STYLE GATES  (why a technique never reached the dice)")
    print("=" * 72)
    armed = [e for e in events if e.get("event") == "armed"]
    with_diag = [a for a in armed if a.get("gated") or a.get("menu")]
    if not with_diag:
        print("  No armed events carry gate attribution yet.")
        print("  Gate stamping landed 2026-07-24 - play a night and re-run.")
        return
    # Only seams where a style was actually ROLLED can answer "was this
    # technique on the menu": a low-confidence or beatless seam takes the
    # fade without ever building one, and counting those would make every
    # style look like it is offered far less often than it is.
    rolled = [a for a in with_diag if a.get("menu")]
    n = len(rolled)
    if not n:
        print(f"  all {len(with_diag)} armed transitions took a forced fade "
              "- no style menu was ever built.")
        return
    played = Counter(a.get("style") for a in rolled)
    offered = Counter()
    gated = defaultdict(Counter)
    for a in rolled:
        for k in (a.get("menu") or {}):
            offered[k] += 1
        for k, why in (a.get("gated") or {}).items():
            gated[k][why] += 1
    styles = sorted(set(offered) | set(gated) | set(played),
                    key=lambda k: -(offered.get(k, 0)))
    print(f"  {n} of {len(with_diag)} armed transitions rolled a style "
          f"({len(with_diag) - n} took a forced fade)\n")
    print(f"  {'style':18} {'on menu':>8} {'played':>7}   top gate")
    for k in styles:
        top = gated[k].most_common(1)
        top_s = (f"{top[0][0]} ({100.0*top[0][1]/n:.0f}%)" if top else "-")
        print(f"  {k:18} {100.0*offered.get(k,0)/n:7.0f}% "
              f"{played.get(k,0):7d}   {top_s}")
    print()
    for k in styles:
        if offered.get(k, 0) and not played.get(k, 0):
            print(f"  {k}: reaches the menu {100.0*offered[k]/n:.0f}% of the "
                  "time and never wins - a WEIGHTS problem, not a gate.")
        elif not offered.get(k, 0):
            print(f"  {k}: never reaches the menu at all - "
                  f"gated by {', '.join(w for w, _ in gated[k].most_common(2))
                              or 'nothing logged'}.")
    print("\n  GATE FREQUENCY (share of armed transitions each gate fired)")
    allg = Counter()
    for k in gated:
        for why, c in gated[k].items():
            allg[why] += c
    for why, c in allg.most_common():
        who = ", ".join(sorted(k for k in gated if why in gated[k]))
        print(f"    {why:22} {c:4d}   -> {who}")


def report_skips(nights):
    print()
    print("=" * 72)
    print("SKIPS  (the operator's loudest, least-recorded opinion)")
    print("=" * 72)
    plays = Counter()
    skips = Counter()
    reoffered = Counter()
    for _, evs in nights:
        seen_skipped = set()
        for e in evs:
            k, t = e.get("event"), e.get("track")
            if k == "play" and t:
                plays[t] += 1
                if t in seen_skipped:
                    reoffered[t] += 1
            elif k == "skip" and t:
                skips[t] += 1
                seen_skipped.add(t)
    if not skips:
        print("  no skips logged")
        return
    print(f"  {sum(skips.values())} skips over {sum(plays.values())} plays "
          f"({100.0*sum(skips.values())/max(sum(plays.values()),1):.0f}%)")
    print(f"\n  {'track':44} {'played':>6} {'skipped':>7} {'rate':>6}")
    for t, k in skips.most_common(15):
        p = plays.get(t, 0)
        print(f"  {t[:44]:44} {p:6d} {k:7d} "
              f"{(100.0*k/p if p else 0):5.0f}%")
    if reoffered:
        print("\n  RE-OFFERED AFTER A SKIP, SAME NIGHT:")
        for t, k in reoffered.most_common(10):
            print(f"    {t[:56]:56} {k}x")
        print("  Each of these is the brain forgetting a rejection it was")
        print("  told about. Skip memory lives in RAM (Brain._recent_skips)")
        print("  and is not persisted, so it also resets every restart.")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Read the DJ's night logs.")
    ap.add_argument("--last", action="store_true", help="most recent night only")
    ap.add_argument("--since", type=int, metavar="N",
                    help="the last N night files")
    ap.add_argument("--terms", action="store_true",
                    help="selection-term validation against measured seams")
    ap.add_argument("--gates", action="store_true",
                    help="which gates keep styles off the menu")
    ap.add_argument("--skips", action="store_true", help="skip analysis")
    ap.add_argument("--all", action="store_true", help="every section")
    ap.add_argument("--log-dir", default=LOG_DIR)
    a = ap.parse_args(argv)

    nights = load_nights(since_days=a.since, last_only=a.last,
                         log_dir=a.log_dir)
    if not nights:
        print(f"no DJ logs in {a.log_dir}")
        return 1
    events = [e for _, evs in nights for e in evs]
    seams = [s for _, evs in nights for s in pair_seams(evs)]

    report_nights(nights)
    report_seams(seams)
    if a.terms or a.all:
        report_terms(seams)
    if a.gates or a.all:
        report_gates(seams, events)
    if a.skips or a.all:
        report_skips(nights)
    if not (a.terms or a.gates or a.skips or a.all):
        print("\n  (--terms / --gates / --skips / --all for the rest)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
