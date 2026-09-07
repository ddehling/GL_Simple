"""Learn from many songs: every ingested SongScript is a point in lever
space. Grouped by style, the scripts give data-derived presets - tempo
range, swing, per-section energy / density / brightness, the layers a
section usually carries, the chord loops that recur - written to
lib/gen/composer/data/learned_styles.json and applied as an overlay by
styles.get_style (GEN_LEARNED=0 disables it).

    presets = derive(scripts)                 # {style: {...}}
    save(presets); load()
    style = apply(style_name, style_dict)     # what get_style calls

A learned preset never replaces a style's slots or grammar; it adjusts
the numbers a style already has (bpm range, swing, section energies,
density) and adds the observed progressions to the style's list, so a
night still plays like the style, tuned toward the songs you fed it."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "composer", "data", "learned_styles.json")
_cache = None


def derive(scripts: list, min_songs: int = 1) -> dict:
    """scripts: list of normalized SongScripts (with "style")."""
    by_style = defaultdict(list)
    for sc in scripts:
        if sc.get("style") and sc.get("sections"):
            by_style[sc["style"]].append(sc)
    out = {}
    for style, scs in by_style.items():
        if len(scs) < min_songs:
            continue
        bpms = [float(sc["bpm"]) for sc in scs if sc.get("bpm")]
        swings = [float(e["swing"]) for sc in scs for e in sc["sections"] if e.get("swing") is not None]
        per_kind = defaultdict(lambda: defaultdict(list))
        layers = defaultdict(Counter)
        progs = Counter()
        for sc in scs:
            for e in sc["sections"]:
                k = e["section"]
                for lever in ("energy", "density", "brightness"):
                    if e.get(lever) is not None:
                        per_kind[k][lever].append(float(e[lever]))
                per_kind[k]["bars"].append(int(e["bars"]))
                for sl in e.get("layers") or []:
                    layers[k][sl] += 1
                if e.get("chords"):
                    from lib.gen.script import chord_deg
                    progs[tuple(chord_deg(x) % 7 for x in e["chords"][:4])] += 1
        sections = {}
        for k, levers in per_kind.items():
            n = len(levers["bars"])
            sec = {"n": n, "bars": [int(np.percentile(levers["bars"], 25)), int(np.percentile(levers["bars"], 75))]}
            for lever in ("energy", "density", "brightness"):
                if levers[lever]:
                    sec[lever] = round(float(np.median(levers[lever])), 3)
            if layers[k]:
                sec["layers"] = [sl for sl, c in layers[k].most_common() if c >= 0.5 * n]
            sections[k] = sec
        out[style] = {"songs": len(scs), "bpm": [round(float(np.percentile(bpms, 10)), 1), round(float(np.percentile(bpms, 90)), 1)] if bpms else None,
                      "swing": round(float(np.median(swings)), 3) if swings else None,
                      "sections": sections,
                      "progressions": [list(p) for p, c in progs.most_common(6) if c >= 1]}
    return out


def save(presets: dict, path: str = PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(presets, fh, indent=1)
    global _cache
    _cache = None
    return path


def load(path: str = PATH) -> dict:
    global _cache
    if _cache is None:
        try:
            with open(path, encoding="utf-8") as fh:
                _cache = json.load(fh)
        except Exception:
            _cache = {}
    return _cache


def apply(name: str, style: dict) -> dict:
    """Overlay the learned preset for `name` onto a style dict (in place)."""
    if os.environ.get("GEN_LEARNED", "1") == "0":
        return style
    pre = load().get(name)
    if not pre:
        return style
    if pre.get("bpm") and pre["bpm"][1] > pre["bpm"][0]:
        lo, hi = pre["bpm"]
        style["bpm"] = (max(40.0, lo - 1.0), min(200.0, hi + 1.0))
    if pre.get("swing") is not None:
        style["swing"] = float(np.clip(pre["swing"], 0.0, 0.33))
    for k, sec in (pre.get("sections") or {}).items():
        if k in style["sections"] and sec.get("energy") is not None and sec.get("n", 0) >= 2:
            style["sections"][k]["energy"] = float(np.clip(sec["energy"], 0.05, 1.0))
        if k in style["sections"] and sec.get("bars") and sec["n"] >= 2:
            lo, hi = sec["bars"]
            style["sections"][k]["bars"] = (max(4, int(lo)), max(4, int(hi)))
    for prog in pre.get("progressions") or []:
        if len(prog) == 4 and prog not in style["progressions"]:
            style["progressions"].append(list(prog))
    style["learned"] = {"songs": pre.get("songs", 0)}
    return style
