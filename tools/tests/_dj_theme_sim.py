"""THEME DISTINGUISHABILITY SIM: do the themes actually play different music?

Same idea as _dj_persona_sim, one level up. A persona is HOW the DJ mixes;
a theme is WHAT it reaches for. Two themes that draw the same tracks in the
same order are one theme with two names, and the operator picking between
them is choosing nothing.

Runs the REAL Brain against the real library for N nights per theme and
reports, per theme, the character of what got played (tempo, energy, the
library-ranked character axes, vocal share) plus the pairwise TRACK-SET
OVERLAP between themes. Overlap is the headline: a Jaccard of 0.5 between
two themes means half of what they play is the same records.

This exists because the 2026-07-24 measurement found nine themes behaving
like about five, and the web picker was cut to five in response. Cutting
the menu was the honest short-term move; this is the instrument for fixing
the cause. Run it after touching anything in themes.py.

Usage:
    python tools/tests/_dj_theme_sim.py                     # all themes, 3x4h
    python tools/tests/_dj_theme_sim.py --nights 2 --hours 3
    python tools/tests/_dj_theme_sim.py --themes groove,peak_heavy,hard_drive
"""
import argparse
import os
import sys
import time as _wall
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

import lib.dj.brain as brain_mod
from lib.dj import resolve_music_dir
from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from lib.dj.themes import BUILTIN_THEMES, adapt_theme, get_theme


class SimClock:
    t = 0.0


def _draw_exit(brain, heat):
    """DJSystem._draw_exit replicated - themes differ in play length too,
    and a theme that plays 4-minute records is a different night from one
    that plays 8-minute records even with identical selection."""
    theme = brain.theme
    span = max(theme.max_play_s - theme.min_play_s, 1.0)
    frac = min(1.0, brain.rng.random() * (1.0 - 0.55 * heat)
               + 0.25 * (1.0 - heat))
    return (theme.min_play_s + frac * span) * brain.persona.play_len_x


def run_night(library, theme_name, hours, seed):
    brain = Brain(library, get_theme(theme_name), seed=seed)
    SimClock.t = 0.0
    total = hours * 3600.0
    cur = brain.choose_first(brain.theme.arc_target(0.0), now=0.0)
    if cur is None:
        return []
    brain.note_played(cur, when=0.0)
    picks = [cur]
    while SimClock.t < total:
        arc = brain.theme.arc_target(SimClock.t / total)
        SimClock.t += _draw_exit(brain, arc)
        cand, meta = brain.choose_next(cur, arc, cur.bpm, now=SimClock.t)
        if cand is None:
            break
        plan = brain.plan_transition(cur, cand, meta, arc=arc)
        SimClock.t += plan.get("beats", 0) * 60.0 / max(cur.bpm, 60.0)
        brain.note_played(cand, when=SimClock.t)
        picks.append(cand)
        cur = cand
    return picks


def profile(picks):
    """What this theme's night SOUNDED like, in library-relative terms.
    Ranked axes (not raw) throughout: raw hardness/hypnotic compress into
    the top of their range on any real collection, so a raw mean says more
    about the analyzer than about the theme."""
    if not picks:
        return {}
    def rank(t, axis):
        v = t.axes_rank.get(axis)
        return 0.5 if v is None else float(v)
    vocals = [t.axes.get("vocal") or 0.0 for t in picks]
    return {
        "n": len(picks),
        "bpm": float(np.mean([t.bpm for t in picks])),
        "energy": float(np.mean([t.energy_proxy() for t in picks])),
        "hardness": float(np.mean([rank(t, "hardness") for t in picks])),
        "hypnotic": float(np.mean([rank(t, "hypnotic") for t in picks])),
        "vocal": float(np.mean(vocals)),
        "vocal_pct": 100.0 * sum(1 for v in vocals if v > 0.35) / len(picks),
    }


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def audit_levers(library, names):
    """Is each theme's flavor lever CONNECTED TO ANYTHING?

    A prefer/avoid tag only steers if that word exists inside the theme's
    own tempo window - and this has silently broken twice. Once because
    the words were invented rather than taken from the library vocabulary
    ('driving'/'mellow' as authored guesses), and once because ten NaN
    energy axes poisoned the percentile that produced those very tags, so
    they vanished from all 649 tracks while the theme kept asking for
    them. Both times the theme just quietly stopped having an opinion.

    A tag on more than ~80% of the pool is equally useless in the other
    direction: preferring something everything already has is a constant.
    """
    bpms = [t.bpm for t in library if t.bpm]
    rows = []
    for name in names:
        th = adapt_theme(get_theme(name), bpms)
        lo, hi = th.bpm_range
        pool = [t for t in library if lo * 0.93 <= t.bpm <= hi * 1.07]
        if not pool:
            continue
        for kind, tags in (("prefer", th.prefer_tags),
                           ("avoid", th.avoid_tags)):
            for tag, w in (tags or {}).items():
                n = sum(1 for t in pool if tag in set(t.all_tags))
                frac = n / len(pool)
                if frac < 0.03:
                    rows.append((name, kind, tag, w, n, len(pool), "DEAD"))
                elif frac > 0.80:
                    rows.append((name, kind, tag, w, n, len(pool),
                                 "near-universal"))
    if not rows:
        print("\nflavor levers: all live (every tag exists, none universal)")
        return
    print("\nDEAD OR USELESS FLAVOR LEVERS")
    for name, kind, tag, w, n, tot, why in rows:
        print(f"  {name:16} {kind:6} {tag:14} w={w:.1f}  "
              f"{n}/{tot} of its tempo pool   <- {why}")
    print("  A dead tag is a theme silently having no opinion. Either the")
    print("  word is not in the library vocabulary, or the stored tags are")
    print("  stale: `python tools/dj/dj_scan.py --retag` re-derives them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default="")
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--nights", type=int, default=3)
    ap.add_argument("--themes", default=",".join(BUILTIN_THEMES))
    args = ap.parse_args()

    brain_mod.time = types.SimpleNamespace(time=lambda: SimClock.t)
    db = LibraryDB(resolve_music_dir(args.music))
    library = [t for t in load_library(db) if not t.excluded]
    names = [n.strip() for n in args.themes.split(",") if n.strip()]
    print(f"library: {len(library)} playable | {args.nights} nights x "
          f"{args.hours:.0f}h per theme | {len(names)} themes")
    audit_levers(library, names)
    print()

    played, prof = {}, {}
    for name in names:
        w0 = _wall.time()
        picks = []
        for night in range(args.nights):
            picks.extend(run_night(library, name, args.hours,
                                   seed=1000 * night + 17))
        played[name] = [t.id for t in picks]
        prof[name] = profile(picks)
        print(f"  {name}: {len(picks)} picks in {_wall.time() - w0:.0f}s")

    cols = ["bpm", "energy", "hardness", "hypnotic", "vocal", "vocal_pct"]
    print(f"\n{'theme':16} {'picks':>6} " + " ".join(f"{c:>9}" for c in cols))
    for name in names:
        p = prof[name]
        if not p:
            continue
        print(f"{name:16} {p['n']:6d} "
              + " ".join(f"{p[c]:9.3f}" for c in cols))

    print("\nTRACK-SET OVERLAP (Jaccard; 1.00 = identical record boxes)")
    print(f"{'':16}" + "".join(f"{n[:8]:>9}" for n in names))
    worst = []
    for a in names:
        row = f"{a:16}"
        for b in names:
            j = 1.0 if a == b else jaccard(played[a], played[b])
            row += f"{j:9.2f}"
            if a < b:
                worst.append((j, a, b))
        print(row)

    worst.sort(reverse=True)
    print("\nmost-overlapping pairs:")
    for j, a, b in worst[:6]:
        flag = "  <- effectively the same theme" if j >= 0.45 else ""
        print(f"  {a:16} / {b:16} {j:.2f}{flag}")
    dup = sum(1 for j, _, _ in worst if j >= 0.45)
    print(f"\n{len(names)} themes, {dup} pairs at/above 0.45 overlap.")
    if dup:
        print("Themes that share half their record box are one choice with")
        print("two labels - widen their bpm percentiles, axis targets or")
        print("tag leans, or fold one into the other.")


if __name__ == "__main__":
    main()
