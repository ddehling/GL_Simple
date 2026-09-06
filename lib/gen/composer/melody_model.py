"""Statistical melody model: motifs sampled from what real melodies do.

Loads lib/gen/composer/data/melody_model.json (built by
tools/gen/melody_corpus.py from public-domain scores). A motif is a
two-bar rhythm cell drawn from the corpus (filtered by density), a pitch
line walked with an order-2 interval model conditioned on metric
strength, a start degree and a cadence degree from the corpus tables,
and a contour drawn from the corpus shapes. Rules the corpus implies are
enforced on top: a leap resolves by step in the opposite direction, the
line stays in a tenth, the last note of a phrase-ending motif is a
cadence tone (tonic, third or fifth).

Without the model file the old random walk is used, so nothing depends
on the data being present. Deterministic under the composer's rng."""
from __future__ import annotations

import json
import os
import random

_MODEL = None
_TRIED = False
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "melody_model.json")
CADENCE_OK = (0, 2, 4)                 # tonic, third, fifth


def load(path=PATH):
    global _MODEL, _TRIED
    if _MODEL is not None or _TRIED:
        return _MODEL
    _TRIED = True
    try:
        with open(path, encoding="utf-8") as fh:
            m = json.load(fh)
        # index the interval table: (prev, strength) -> [(next, count)]
        table = {}
        for prev, strength, nxt, cnt in m.get("intervals", []):
            table.setdefault((int(prev), int(strength)), []).append((int(nxt), int(cnt)))
        m["_table"] = table
        m["_rhythms"] = [(tuple(int(s) for s in steps), int(c)) for steps, c in m.get("rhythms", [])]
        _MODEL = m
    except Exception:
        _MODEL = None
    return _MODEL


def available():
    return load() is not None


def _weighted(rng: random.Random, pairs):
    tot = sum(c for _, c in pairs)
    r = rng.random() * tot
    for v, c in pairs:
        r -= c
        if r <= 0:
            return v
    return pairs[-1][0]


def _weighted_dict(rng, d, default):
    pairs = [(int(k), int(v)) for k, v in d.items()]
    return _weighted(rng, pairs) if pairs else default


def sample_rhythm(rng: random.Random, n_notes: int, steps_per_bar: int = 16):
    """A two-bar onset set from the corpus with about n_notes onsets
    (nearest sizes), or None."""
    m = load()
    if not m or not m["_rhythms"]:
        return None
    cands = [(steps, c) for steps, c in m["_rhythms"] if abs(len(steps) - n_notes) <= 1 and max(steps) < 2 * steps_per_bar]
    if not cands:
        cands = sorted(m["_rhythms"], key=lambda sc: abs(len(sc[0]) - n_notes))[:20]
    steps = _weighted(rng, cands)
    return list(steps)


def sample_line(rng: random.Random, steps, cadence: bool = False, lo: int = -4, hi: int = 9):
    """Degree offsets (from the chord root) for the given steps, walked
    with the corpus interval model. Returns a list the length of steps."""
    m = load()
    if not m:
        return None
    table = m["_table"]
    start = _weighted_dict(rng, m.get("start_degrees", {}), 0)
    cur = start if start <= 4 else start - 7            # keep the start near the root
    degs = [cur]
    prev = _weighted_dict(rng, m.get("first_interval", {}), 0)
    for i in range(1, len(steps)):
        s = steps[i]
        strength = 2 if s % 4 == 0 else (1 if s % 2 == 0 else 0)
        pairs = table.get((prev, strength)) or table.get((0, strength)) or [(0, 1), (1, 1), (-1, 1)]
        iv = _weighted(rng, pairs)
        if abs(prev) >= 3 and rng.random() < 0.85:
            iv = (-1 if prev > 0 else 1) * rng.choice((1, 1, 2))   # a leap resolves by step the other way
        nxt = cur + iv
        if nxt > hi or nxt < lo:
            iv = -iv
            nxt = cur + iv
        cur = max(lo, min(hi, nxt))
        degs.append(cur)
        prev = iv
    if cadence and degs:
        target = _weighted_dict(rng, m.get("cadence_degrees", {}), 0)
        if target % 7 not in CADENCE_OK:
            target = 0
        # end on the cadence tone nearest the line's last note
        cands = [target + 7 * k for k in (-2, -1, 0, 1) if lo <= target + 7 * k <= hi] or [max(lo, min(hi, target))]
        degs[-1] = min(cands, key=lambda d: abs(d - degs[-1]))
    return degs


def sample_contour(rng: random.Random):
    m = load()
    if not m or not m.get("contours"):
        return "flat"
    names = {"arch": "arch", "rise": "rise", "fall": "fall", "valley": "wave", "flat": "flat"}
    shape = _weighted(rng, [(k, int(v)) for k, v in m["contours"].items()])
    return names.get(shape, "flat")


def interval_hist():
    m = load()
    return {int(k): float(v) for k, v in (m or {}).get("interval_hist", {}).items()}
