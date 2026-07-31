"""Night-log analysis: the DATA layer behind tools/dj/dj_review.py.

The live system self-assesses every seam it plays (`seam_quality` events)
and stamps the prediction that produced it (`armed` events). This module
loads those logs and joins/aggregates them into plain data structures, so
BOTH the review CLI (human-formatted) and the planner (post-mortem tab,
seam badges) read the same evidence with the same thresholds.

Keep every threshold here in one place: a seam is "rough" by the ENGINE's
own bar - the same numbers system._assess_seam uses to hand out an auto
thumbs-down - so no reader ever disagrees with the learning loop about
what a bad seam was.
"""
import glob
import json
import math
import os
import statistics as st
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
LOG_DIR = os.path.join(_ROOT, "logs")

FLAM_BEATS = 0.12                # verdict bar (matches system._assess_seam)
HOLE_S = 1.5
# ...but the codebase's own stated audibility threshold for two percussive
# transients is ~25 ms (brain.plan_transition's groove-offset gate), which
# is ~0.055 beats at 128 bpm - less than HALF the verdict bar. The two
# numbers measure slightly different things (grid-phase error vs kick
# placement); a seam in the band between them is not proven audible, but
# it IS invisible to the learning loop - count it separately.
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


def corr(xs, ys):
    """Pearson r over the pairs where both are present; None on thin or
    degenerate samples."""
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


# ------------------------------------------------------------ aggregation

def night_summary(nights):
    """One dict per night: what played, how long, how it went."""
    out = []
    for date, evs in nights:
        c = Counter(e.get("event") for e in evs)
        seams = [e for e in evs if e.get("event") == "seam_quality"]
        out.append({
            "date": date,
            "hours": max((float(e.get("clock_s") or 0.0)
                          for e in evs), default=0.0) / 3600.0,
            "plays": c.get("play", 0),
            "seams": len(seams),
            "rough": sum(1 for e in seams if is_rough(e)),
            "skips": c.get("skip", 0),
            "aborts": c.get("abort", 0),
            "bailouts": c.get("flam_bailout", 0),
            "themes": list(dict.fromkeys(
                e.get("theme") for e in evs
                if e.get("event") == "theme" and e.get("theme"))),
        })
    return out


def night_tracks(events):
    """The night's play-by-play: [(title, artist, via_style)], in order."""
    out = []
    for e in events:
        if e.get("event") == "play":
            out.append((e.get("track") or "?", e.get("artist") or "",
                        e.get("via") or "start"))
    return out


def night_seam_rows(events):
    """Per-seam verdict rows for one night, in play order: dicts with
    a, b, style, verdict, max_err_beats, hole_s, urgent."""
    rows = []
    for s in pair_seams(events):
        q = s["q"]
        rows.append({
            "a": q.get("a") or "?", "b": q.get("b") or "?",
            "style": q.get("style") or "?",
            "verdict": q.get("verdict") or "?",
            "max_err_beats": float(q.get("max_err_beats") or 0.0),
            "hole_s": float(q.get("hole_s") or 0.0),
            "urgent": bool(q.get("urgent")),
            "rough": is_rough(q),
        })
    return rows


def pair_verdicts(nights):
    """Cross-night seam history keyed by TITLE pair (seam_quality logs
    titles, not ids): {(a_title, b_title): [{date, style, verdict,
    max_err_beats, hole_s, urgent, rough}, ...]} oldest first.

    This is the planner's badge source: "this exact pairing flammed on
    07-12" next to tonight's plan - visibility for evidence the pair-memory
    multiplier already absorbed numerically."""
    out = defaultdict(list)
    for date, evs in nights:
        for s in pair_seams(evs):
            q = s["q"]
            a, b = q.get("a"), q.get("b")
            if not a or not b:
                continue
            out[(a, b)].append({
                "date": date, "style": q.get("style") or "?",
                "verdict": q.get("verdict") or "?",
                "max_err_beats": float(q.get("max_err_beats") or 0.0),
                "hole_s": float(q.get("hole_s") or 0.0),
                "urgent": bool(q.get("urgent")),
                "rough": is_rough(q),
            })
    return dict(out)
