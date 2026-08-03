"""PERSONA DISTINGUISHABILITY SIM: brain-level simulated nights, no audio.

Runs the REAL Brain (selection scoring, style menu, all gates) against the
real library for N nights per persona and prints the per-persona style /
pacing / harmonic histograms. This is the acceptance test for the persona
layer: if two personas' nights are statistically indistinguishable, the
multipliers are too timid; if `neutral` drifts from pre-persona behavior,
the identity contract is broken.

No decks, no submix, no rendering - a night is simulated as the seam
decisions the brain would make, with the arc-coupled play-length draw
replicated from DJSystem._draw_exit. Simulated time drives the moment
cooldown via a module-time shim.

Usage:
    python tools/tests/_dj_persona_sim.py                       # 6h x 6 nights each
    python tools/tests/_dj_persona_sim.py --nights 8 --hours 4
    python tools/tests/_dj_persona_sim.py --personas neutral,monk,showman
"""
import argparse
import json
import os
import sys
import time as _wall
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

import lib.dj.brain as brain_mod
from lib.dj.brain import Brain, camelot_compat, load_library
from lib.dj.db import LibraryDB
from lib.dj.persona import PERSONAS
from lib.dj.themes import get_theme

MUSIC = "C:/Users/ddehl/Desktop/Devel/music"

SPECTACLE = ("loop_build", "cut_at_drop", "stem_drum_swap", "acapella_out")
PUNCHY = ("cut_at_drop", "loop_build", "loop_roll_exit",
          "echo_out", "stem_drum_swap")


class SimClock:
    t = 0.0


def _draw_exit(brain, heat):
    """DJSystem._draw_exit replicated (same rng stream, same arc coupling,
    same persona pacing multiplier)."""
    theme = brain.theme
    span = max(theme.max_play_s - theme.min_play_s, 1.0)
    frac = min(1.0, brain.rng.random() * (1.0 - 0.55 * heat)
               + 0.25 * (1.0 - heat))
    return (theme.min_play_s + frac * span) * brain.persona.play_len_x


def run_night(library, theme_name, persona, hours, seed):
    theme = get_theme(theme_name)
    brain = Brain(library, theme, seed=seed)
    brain.persona = persona
    SimClock.t = 0.0
    total = hours * 3600.0
    cur = brain.choose_first(brain.theme.arc_target(0.0), now=0.0)
    if cur is None:
        return []
    brain.note_played(cur, when=0.0)
    seams = []
    while SimClock.t < total:
        arc = brain.theme.arc_target(SimClock.t / total)
        play_s = _draw_exit(brain, arc)
        SimClock.t += play_s
        cand, meta = brain.choose_next(cur, arc, cur.bpm, now=SimClock.t)
        if cand is None:
            break
        plan = brain.plan_transition(cur, cand, meta, arc=arc)
        kc = camelot_compat(cur.camelot, cand.camelot)
        v = cand.axes.get("vocal", 0.5)
        seams.append({
            "style": plan["style"], "beats": plan.get("beats", 0),
            "key_compat": kc, "arc": arc, "play_s": play_s,
            "eff_bpm": (meta or {}).get("eff_bpm", cand.bpm),
            "vocal": v if v is not None else 0.5,
            "moment": plan["style"] in SPECTACLE and arc >= 0.82,
            "clash_fade": bool((meta or {}).get("tempo_clash")),
        })
        SimClock.t += plan.get("beats", 0) * 60.0 / max(cur.bpm, 60.0)
        brain.note_played(cand, when=SimClock.t)
        cur = cand
    return seams


def summarize(name, all_seams, nights):
    n = len(all_seams)
    if not n:
        return {"persona": name, "seams": 0}
    styles = {}
    for s in all_seams:
        styles[s["style"]] = styles.get(s["style"], 0) + 1
    pct = {k: 100.0 * c / n for k, c in sorted(
        styles.items(), key=lambda kv: -kv[1])}
    lb = [s for s in all_seams if s["style"] == "long_blend"]
    kc = np.array([s["key_compat"] for s in all_seams])
    return {
        "persona": name,
        "seams": n,
        "tracks_per_night": round(n / nights, 1),
        "styles_pct": {k: round(v, 1) for k, v in pct.items()},
        "punchy_pct": round(100.0 * sum(
            1 for s in all_seams if s["style"] in PUNCHY) / n, 1),
        "fade_pct": round(100.0 * sum(
            1 for s in all_seams
            if s["style"] in ("long_fade",) or s["clash_fade"]) / n, 1),
        "blend96_pct": round(100.0 * sum(
            1 for s in lb if s["beats"] >= 96) / max(len(lb), 1), 1),
        "mean_blend_beats": round(float(np.mean(
            [s["beats"] for s in all_seams])), 1),
        "mean_play_min": round(float(np.mean(
            [s["play_s"] for s in all_seams])) / 60.0, 2),
        "key_close_pct": round(100.0 * float((kc >= 0.9).mean()), 1),
        "key_clash_pct": round(100.0 * float((kc < 0.55).mean()), 1),
        "bpm_std": round(float(np.std(
            [s["eff_bpm"] for s in all_seams])), 1),
        "vocal_mean": round(float(np.mean(
            [s["vocal"] for s in all_seams])), 3),
        "moments_per_night": round(sum(
            1 for s in all_seams if s["moment"]) / nights, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", default=MUSIC)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--nights", type=int, default=6)
    ap.add_argument("--theme", default="all_night")
    ap.add_argument("--personas", default=",".join(PERSONAS))
    ap.add_argument("--json", default="", help="dump per-seam records here")
    args = ap.parse_args()

    # Simulated time drives the brain's moment cooldown.
    brain_mod.time = types.SimpleNamespace(time=lambda: SimClock.t)

    db = LibraryDB(args.music)
    library = load_library(db)
    print(f"library: {len(library)} playable tracks | theme {args.theme} | "
          f"{args.nights} nights x {args.hours:.0f}h per persona")

    rows, dump = [], {}
    for name in args.personas.split(","):
        persona = PERSONAS[name.strip()]
        w0 = _wall.time()
        seams = []
        for night in range(args.nights):
            seams.extend(run_night(library, args.theme, persona,
                                   args.hours, seed=1000 * night + 17))
        rows.append(summarize(persona.name, seams, args.nights))
        dump[persona.name] = seams
        print(f"  {persona.name}: {len(seams)} seams "
              f"in {_wall.time() - w0:.0f}s wall")

    cols = ["persona", "tracks_per_night", "punchy_pct", "fade_pct",
            "blend96_pct", "mean_blend_beats", "mean_play_min",
            "key_close_pct", "key_clash_pct", "bpm_std", "vocal_mean",
            "moments_per_night"]
    print("\n" + " | ".join(f"{c:>16}" for c in cols))
    for r in rows:
        print(" | ".join(f"{str(r.get(c, '-')):>16}" for c in cols))
    print("\nstyle histograms (% of seams):")
    for r in rows:
        print(f"  {r['persona']:>12}: {r.get('styles_pct', {})}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": rows, "seams": dump}, f)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
