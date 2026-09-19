"""THE SET: a chain of records ahead, and a route to a far one (docs/DJ_THE_SET.md).

Pure functions over the library, the pairing file's records and a few numbers from the room; no engine, no
audio, no Qt. The Director keeps the chain (`horizon`) and prunes it with `hop_ok`; the planner tab shows it.

A hop from record A to candidate C is judged against the clock A leaves behind. With the clock following the
record (RemixConductor.follow_clock) that clock is A's native tempo; with a fixed clock it is the clock itself.
"""
import math
import random

from lib.dj.brain import camelot_compat, _shift_camelot

WALL = (0.90, 1.10)          # the tempo wall of the layered engine (RATE_MIN / RATE_MAX in lib/dj/remix.py)
KEY_MIN = 0.68               # the conductor's ENTRY_FIT: a neighbour key or the relative major/minor, after a shift
#                              (the chain must pass the pick's own gate, or the brain picks instead of the plan)
SHIFTS = (0, 1, -1, 2, -2)   # the deck's transposes (KEY_SHIFTS in remix.py) with their cost per semitone
SHIFT_COST = 0.04
RUNWAY_S = 90.0              # a record needs its minimum play and a seam's runway after its body starts
SEAM_S = 30.0                # a seam in the projected timing (16 bars at ~128)
GENRE_BONUS = 1.08
ENERGY_WEIGHT = 0.3


def key_fit(cam_a, cam_b):
    """(fit, shift): the best Camelot fit of B against A over the deck's shifts, each semitone costing a little."""
    if not cam_a or not cam_b:
        return 0.6, 0
    best = (camelot_compat(cam_a, cam_b), 0)
    for s in SHIFTS[1:]:
        try:
            c = camelot_compat(cam_a, _shift_camelot(cam_b, s)) - SHIFT_COST * abs(s)
        except Exception:
            continue
        if c > best[0] + 1e-9:
            best = (c, s)
    return best


def _energy(t):
    try:
        return float(t.energy_proxy())
    except Exception:
        return 0.5


def _body_start(t):
    secs = getattr(t, "sections", None) or []
    for s in secs:
        if s.get("kind") not in ("intro",):
            return float(s.get("start_s") or 0.0)
    return 0.0


def _genres(rec, t):
    g = set(x.lower() for x in (rec.get("genres") or []) if x)
    gg = getattr(t, "genre", None)
    if gg:
        g.add(str(gg).lower())
    return g


def hop_ok(prev, cand, clock_bpm, played=(), avoid=(), play_min_s=210.0, need_stems=True):
    """Can `cand` follow `prev` from `clock_bpm`? None when it can, else the reason (a short phrase)."""
    if cand is None or not getattr(cand, "bpm", None):
        return "no tempo"
    if prev is not None and cand.id == prev.id:
        return "the same song"
    if cand.id in played:
        return "played tonight"
    if cand.id in avoid:
        return "not tonight"
    if need_stems and not getattr(cand, "has_stems", False):
        return "no stems"
    rate = float(clock_bpm) / float(cand.bpm)
    if not (WALL[0] <= rate <= WALL[1]):
        return f"outside the wall ({float(cand.bpm):.0f} against {float(clock_bpm):.0f})"
    if (float(getattr(cand, "duration_s", 0.0) or 0.0) - _body_start(cand)) / rate < float(play_min_s) + RUNWAY_S:
        return "too short for a record"
    if prev is not None:
        kf, _s = key_fit(getattr(prev, "camelot", None), getattr(cand, "camelot", None))
        if kf < KEY_MIN:
            return f"a key clash ({getattr(cand, 'camelot', '?')} against {getattr(prev, 'camelot', '?')})"
    return None


LEAN_STEP = 0.04             # the TEMPO dial: "faster" asks each record to sit about 4 % above the one before


def pair_score(prev, cand, clock_bpm, arc_e, recs, seed=0, lean=0.0):
    """How well `cand` follows `prev` as the next record: key × tempo closeness (to the step the TEMPO dial
    asks for: `lean` -1 / 0 / +1) × the arc's energy × how it can arrive (the pairing file's seam word) × kin.
    Returns (score, words, shift)."""
    rate = float(clock_bpm) / float(cand.bpm)
    kf, shift = key_fit(getattr(prev, "camelot", None), getattr(cand, "camelot", None)) if prev is not None else (0.8, 0)
    step = (1.0 / rate - 1.0) - float(lean) * LEAN_STEP           # the record's tempo against the one asked for
    tempo = max(0.0, 1.0 - abs(step) / 0.1 * 0.3)
    e1 = _energy(cand)
    en = 1.0 - ENERGY_WEIGHT * min(1.0, abs(e1 - float(arc_e)) / 0.5)
    rec = recs.get(str(cand.id)) or {}
    sm = rec.get("seam") or {}
    ways = [w for w, ok in (("drums first", sm.get("drums")), ("bass swap", sm.get("bass"))) if ok]
    arrive = 1.0 if ways else (0.85 if sm.get("after", True) else 0.6)
    kin = 1.0
    if prev is not None:
        gp = _genres(recs.get(str(prev.id)) or {}, prev)
        gc = _genres(rec, cand)
        if gp and gc and (gp & gc):
            kin = GENRE_BONUS
    score = kf * tempo * en * arrive * kin
    # a small seeded jitter per pair: the same night gives the same chain, another night another one
    if seed:
        score *= 1.0 + 0.02 * (random.Random(hash((seed, cand.id)) & 0xFFFFFFFF).random() - 0.5)
    e0 = _energy(prev) if prev is not None else float(arc_e)
    words = [f"{float(cand.bpm):.0f} ({100 * (1.0 / rate - 1.0):+.0f} %)" if abs(rate - 1.0) >= 0.005 else f"{float(cand.bpm):.0f} (same tempo)",
             f"{getattr(cand, 'camelot', None) or '?'} {kf:.2f}" + (f" shift {shift:+d}" if shift else ""),
             ("same energy" if abs(e1 - e0) < 0.12 else ("hotter" if e1 > e0 else "calmer")),
             (" or ".join(ways) if ways else ("the cut" if sm else ""))]
    return score, [w for w in words if w], shift


def chain(cur, library, recs, clock_bpm, arc_fn, played=(), avoid=(), busy=(), play_min_s=210.0, n=5,
          width=4, follow=True, t0=0.0, seed=0, pinned=(), lean=0.0):
    """Up to `n` records after `cur` (and after the `pinned` ones, in order: the user's queue), each chosen
    against the one before it. `arc_fn(t_s)` is the arc's energy target `t_s` seconds from now. Returns a list
    of hops: {id, title, artist, bpm, camelot, score, why (words), shift, t (projected start, s from now),
    clock (the clock after it)}. A beam search of `width`, deterministic for a seed."""
    used = set(played) | set(avoid) | set(busy)
    prev, clock, t = cur, float(clock_bpm), float(t0)
    by_id = {x.id: x for x in library}
    if cur is not None:
        used.add(cur.id)
    step_s = float(play_min_s) + SEAM_S
    # the pinned ones first: they are the user's, judged not chosen
    for pid in pinned:
        p = by_id.get(int(pid))
        if p is None:
            continue
        used.add(p.id)
        prev, t = p, t + step_s
        clock = float(p.bpm) if (follow and p.bpm) else clock
    if prev is None:
        return []
    beam = [(0.0, [], prev, clock, t)]
    for _depth in range(n):
        nxt = []
        for total, hops, pv, ck, tt in beam:
            taken = used | {h["id"] for h in hops}
            arc_e = float(arc_fn(tt + step_s)) if arc_fn else 0.6
            cands = []
            for c in library:
                if c.id in taken or hop_ok(pv, c, ck, play_min_s=play_min_s) is not None:
                    continue
                sc, words, shift = pair_score(pv, c, ck, arc_e, recs, seed=seed, lean=lean)
                cands.append((sc, c, words, shift))
            cands.sort(key=lambda x: (-x[0], x[1].id))
            for sc, c, words, shift in cands[:max(2, width)]:
                hop = {"id": c.id, "title": c.title, "artist": getattr(c, "artist", "") or "", "bpm": float(c.bpm),
                       "camelot": getattr(c, "camelot", None), "score": round(sc, 3), "why": words, "shift": shift,
                       "t": tt + step_s, "clock": float(c.bpm) if follow else ck, "energy": round(_energy(c), 2)}
                nxt.append((total + sc, hops + [hop], c, hop["clock"], tt + step_s))
        if not nxt:
            break
        nxt.sort(key=lambda x: (-x[0], [h["id"] for h in x[1]]))
        beam = nxt[:width]
    return beam[0][1] if beam else []


def route(cur, dest, library, recs, clock_bpm, played=(), avoid=(), busy=(), play_min_s=210.0, max_via=3,
          follow=True, width=8):
    """A chain of at most `max_via` records from `cur` to `dest`, each hop inside the wall and fitting in key,
    heading toward `dest`'s tempo. Returns the intermediates as hops (see `chain`) - an empty list when `dest`
    follows `cur` directly - or None when nothing reaches it. Only makes sense when the clock follows the record;
    with a fixed clock a song outside the wall stays outside it."""
    if cur is None or dest is None or not dest.bpm:
        return None
    if hop_ok(cur, dest, clock_bpm, played=(), avoid=(), play_min_s=play_min_s) is None:
        return []
    if not follow:
        return None
    used = set(played) | set(avoid) | set(busy) | {cur.id, dest.id}
    goal = math.log(float(dest.bpm))
    beam = [(0.0, [], cur, float(clock_bpm))]
    for _depth in range(max_via):
        nxt, done = [], []
        for total, hops, pv, ck in beam:
            taken = used | {h["id"] for h in hops}
            here = abs(math.log(float(ck)) - goal)
            cands = []
            for c in library:
                if c.id in taken or hop_ok(pv, c, ck, play_min_s=play_min_s) is not None:
                    continue
                # every hop must make headway toward the destination's tempo (a fifth of the wall at least)
                left = abs(math.log(float(c.bpm)) - goal)
                if left > here - 0.02:
                    continue
                sc, words, shift = pair_score(pv, c, ck, 0.6, recs)
                cands.append((sc - 6.0 * left, sc, c, words, shift))
            cands.sort(key=lambda x: (-x[0], x[2].id))
            for _h, sc, c, words, shift in cands[:width]:
                hop = {"id": c.id, "title": c.title, "artist": getattr(c, "artist", "") or "", "bpm": float(c.bpm),
                       "camelot": getattr(c, "camelot", None), "score": round(sc, 3), "why": words, "shift": shift, "t": None,
                       "clock": float(c.bpm), "energy": round(_energy(c), 2), "via": True}
                path = hops + [hop]
                if hop_ok(c, dest, float(c.bpm), play_min_s=play_min_s) is None:
                    done.append((total + sc, path))
                else:
                    nxt.append((total + sc, path, c, float(c.bpm)))
        if done:
            done.sort(key=lambda x: (-x[0] / len(x[1]), len(x[1])))
            return done[0][1]
        if not nxt:
            return None
        nxt.sort(key=lambda x: -x[0])
        beam = nxt[:width]
    return None


def hops_by_tempo(clock_bpm, bpms, max_hops=3):
    """For every tempo in `bpms`: how many records it takes to reach it from `clock_bpm` by tempo alone (1 =
    inside the wall), or None beyond `max_hops`. Cheap: the wall widens band by band over the tempos present.
    The SONGS list uses it for the 'via N' word; the real route (key, stems, runway) is asked for on demand."""
    out = {}
    lo, hi = float(clock_bpm) * WALL[0], float(clock_bpm) * WALL[1]
    left = sorted(set(float(b) for b in bpms if b))
    for k in range(1, max_hops + 1):
        inside = [b for b in left if lo <= b <= hi]
        for b in inside:
            out[b] = k
        if not inside:
            break
        left = [b for b in left if b not in out]
        lo, hi = min(inside) * WALL[0], max(inside) * WALL[1]
    return out
