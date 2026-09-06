"""Drum and rhythmic grids: 16 steps per bar, velocities 0..1.

Euclidean patterns give perc parts that sound 'placed' rather than random;
probability grids give hats their life; kicks and claps are idiomatic and
energy-gated so the section masks (form.py) do the big moves and energy
does the small ones.

Rhythmic language (style["drums"]):
  four       four-on-the-floor (house / techno / trance)
  broken     1, and-of-2, 3-ish (downtempo)
  breakbeat  a break chosen per phrase from BREAKS (drum and bass, breaks)
  halftime   kick on 1, snare on 3 (hip-hop, trap-ish) - also forced in
             the sections a style lists under "halftime_in"
Fills come from a library (FILLS), one shape per fill bar, shared by kick,
snare and toms so they play the SAME fill. Perc cells may be 12 steps
long, rolling 3-against-4 across the bar (ctx "poly"). Builds double
the hats (ctx "double").

The kit talks to itself: hats thin out when the bass is busy (ctx
"bass_hits"), the shaker breathes across the bar, toms only play fills,
the ride only rides the drop, the rim fills the backbeat's gaps."""
from __future__ import annotations

import random


def euclid(k: int, n: int, rot: int = 0) -> list:
    """Bjorklund: k hits spread as evenly as possible over n steps."""
    if k <= 0:
        return [0] * n
    if k >= n:
        return [1] * n
    pattern = []
    counts, remainders = [], [k]
    divisor = n - k
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level += 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)

    def build(lvl):
        if lvl == -1:
            pattern.append(0)
        elif lvl == -2:
            pattern.append(1)
        else:
            for _ in range(counts[lvl]):
                build(lvl - 1)
            if remainders[lvl] != 0:
                build(lvl - 2)

    build(level)
    i = pattern.index(1)
    pattern = pattern[i:] + pattern[:i]
    rot %= n
    return pattern[rot:] + pattern[:rot]


TOM_HI, TOM_MID, TOM_LO, TOM_FLOOR = 50, 48, 45, 43

# Classic breaks as 16-step kick / snare / ghost maps (velocities).
BREAKS = {
    "amen":    {"kick": [(0, 1.0), (2, 0.8), (10, 0.9)],        "snare": [(4, 1.0), (7, 0.45), (9, 0.5), (12, 1.0), (15, 0.4)]},
    "funky":   {"kick": [(0, 1.0), (6, 0.7), (10, 0.85)],       "snare": [(4, 1.0), (12, 1.0), (14, 0.35)]},
    "think":   {"kick": [(0, 1.0), (10, 0.9)],                  "snare": [(4, 1.0), (7, 0.4), (12, 1.0), (13, 0.35)]},
    "apache":  {"kick": [(0, 1.0), (3, 0.6), (8, 0.8), (11, 0.6)], "snare": [(4, 1.0), (12, 1.0), (15, 0.4)]},
    "twostep": {"kick": [(0, 1.0), (10, 0.9)],                  "snare": [(4, 1.0), (12, 1.0)]},
}

# Fill library: the last bar of a section. kick/snare: (step, vel);
# tom: (step, vel, midi). One shape is drawn per fill bar.
FILLS = [
    {"kick": [(0, 1.0), (4, 1.0)], "snare": [(8, 0.6), (10, 0.7), (12, 0.8), (14, 0.9)], "tom": []},
    {"kick": [(0, 1.0), (4, 1.0), (8, 0.9)], "snare": [(12, 0.8), (13, 0.6), (14, 0.9), (15, 0.7)], "tom": []},
    {"kick": [(0, 1.0), (4, 1.0)], "snare": [(8, 0.7), (9, 0.5)],
     "tom": [(10, 0.75, TOM_HI), (11, 0.6, TOM_HI), (12, 0.8, TOM_MID), (13, 0.65, TOM_LO), (14, 0.85, TOM_FLOOR), (15, 0.9, TOM_FLOOR)]},
    {"kick": [(0, 1.0), (6, 0.8), (10, 0.9)], "snare": [(4, 0.9), (12, 1.0), (14, 0.6), (15, 0.7)], "tom": []},
    {"kick": [(0, 1.0)], "snare": [(2, 0.5), (4, 0.8), (6, 0.6), (8, 0.9), (10, 0.7), (12, 1.0), (14, 0.8), (15, 0.9)], "tom": []},
    {"kick": [(0, 1.0), (4, 1.0), (8, 1.0)], "snare": [(12, 1.0)],
     "tom": [(9, 0.7, TOM_HI), (10, 0.7, TOM_MID), (11, 0.75, TOM_LO), (13, 0.8, TOM_FLOOR), (14, 0.85, TOM_FLOOR), (15, 0.9, TOM_FLOOR)]},
    {"kick": [(0, 1.0), (4, 1.0), (8, 1.0), (12, 1.0)], "snare": [(14, 0.8), (15, 0.95)], "tom": []},
]


class Drums:
    def __init__(self, style: dict, rng: random.Random):
        self.style = style
        self.rng = rng
        self.steps = style["steps_per_bar"]
        self.kind = style.get("drums") or ("four" if style["bpm"][0] >= 110 else "broken")
        self._perc_rot = rng.randrange(self.steps)
        self._perc_k = rng.choice([3, 5, 7])
        self._perc_len = 16
        self._break = rng.choice(list(BREAKS))
        self._break_phrase = -1
        self._fill = None
        self._fill_key = None

    def _fill_for(self, ctx, bar_in_phrase):
        key = (ctx.get("phrase", 0), bar_in_phrase)
        if self._fill_key != key:
            self._fill_key = key
            self._fill = self.rng.choice(FILLS)
        return self._fill

    def _break_for(self, ctx):
        ph = ctx.get("phrase", 0)
        if ph != self._break_phrase:
            self._break_phrase = ph
            if self.rng.random() < 0.35:
                self._break = self.rng.choice(list(BREAKS))
        return BREAKS[self._break]

    def bar(self, slot: str, energy: float, bar_in_phrase: int, nbars: int,
            last_of_section: bool, ctx: dict | None = None) -> list:
        """Return [(step, vel), ...] ([(step, vel, midi)] for toms) for one
        bar of one drum slot. ctx: bass_hits (int, this bar), kick_steps
        (set), section (str), phrase (int), halftime, double, poly."""
        S = self.steps
        rng = self.rng
        ctx = ctx or {}
        dens = self.style["density"].get(slot, 1.0)
        out = []
        fill = last_of_section and bar_in_phrase == nbars - 1
        section = ctx.get("section", "")
        bass_busy = min(1.0, ctx.get("bass_hits", 0) / 8.0)
        kind = "halftime" if ctx.get("halftime") else self.kind
        double = bool(ctx.get("double"))
        if slot == "kick":
            if fill:
                return list(self._fill_for(ctx, bar_in_phrase)["kick"])
            if kind == "four":
                for q in range(4):
                    out.append((q * S // 4, 1.0))
                if energy > 0.75 and rng.random() < 0.25:
                    out.append((S - 2, 0.6))          # pickup before the bar
            elif kind == "breakbeat":
                out = list(self._break_for(ctx)["kick"])
                if energy > 0.7 and rng.random() < 0.3:
                    out.append((S - 1, 0.55))
            elif kind == "halftime":
                out.append((0, 1.0))
                if rng.random() < 0.5 + 0.4 * energy:
                    out.append((7 if rng.random() < 0.6 else 10, 0.8))
                if energy > 0.6 and rng.random() < 0.3:
                    out.append((13, 0.6))
            else:                                      # broken: 1, and-of-2, 3-ish
                out.append((0, 1.0))
                out.append((S * 5 // 8, 0.85) if rng.random() < 0.7 else (S // 2, 0.9))
                if rng.random() < 0.35 * energy:
                    out.append((S * 7 // 8 + (0 if rng.random() < 0.5 else -1), 0.6))
        elif slot == "snare":
            if fill:
                return list(self._fill_for(ctx, bar_in_phrase)["snare"])
            if kind == "breakbeat":
                out = [(s, v) for s, v in self._break_for(ctx)["snare"] if v >= 0.6 or rng.random() < 0.4 + 0.5 * energy]
            elif kind == "halftime":
                out = [(S // 2, 1.0)]
                if energy > 0.5 and rng.random() < 0.35:
                    out.append((S - 2, 0.4))          # ghost into the 1
            else:
                out = [(S // 4, 0.95), (3 * S // 4, 1.0)]
                if energy > 0.6 and rng.random() < 0.3:
                    out.append((3 * S // 4 + 3, 0.45))    # ghost
        elif slot == "tom":
            if fill:
                return list(self._fill_for(ctx, bar_in_phrase)["tom"])
            if energy > 0.55 and rng.random() < 0.18 * dens:
                out.append((S - 2 if rng.random() < 0.5 else 10, 0.6, rng.choice((TOM_HI, TOM_MID, TOM_LO, TOM_FLOOR))))
        elif slot == "rim":
            # cross-stick colour between the backbeats; more when the snare is quiet
            for s in (3, 7, 11, 14):
                if rng.random() < dens * (0.35 + 0.4 * energy):
                    out.append((s, 0.5 + 0.3 * rng.random()))
        elif slot == "hat":
            base = 1 if (double or energy >= 0.5) else 2   # 8ths below half energy, 16ths above / in builds
            if kind == "halftime" and not double:
                base = 2
            for s in range(0, S, base):
                onbeat = (s % 4 == 0)
                offbeat = (s % 4 == 2)
                p = dens * (0.95 if offbeat else (0.8 if onbeat else 0.45 + 0.5 * energy))
                if double:
                    p = max(p, dens * 0.97)                # builds: every 16th, no gaps
                if not offbeat and not onbeat and not double:
                    p *= 1.0 - 0.35 * bass_busy         # leave room when the bass is talking
                if rng.random() < p:
                    vel = 0.9 if offbeat else (0.55 if onbeat else 0.4 + 0.3 * rng.random())
                    vel *= 0.9 + 0.2 * rng.random()      # no two hits the same
                    out.append((s, min(1.0, vel)))
        elif slot == "shaker":
            # 16ths with a breathing accent shape: strong on 8ths, weak between,
            # a slow swell across the bar so it moves instead of ticking
            for s in range(S):
                p = dens * (0.9 if s % 2 == 0 else 0.55 + 0.35 * energy) * (1.0 - 0.25 * bass_busy)
                if double:
                    p = max(p, dens * 0.9)
                if rng.random() < p:
                    sway = 0.85 + 0.15 * (1.0 - abs((s / S) * 2.0 - 1.0))
                    vel = (0.7 if s % 4 == 0 else (0.5 if s % 2 == 0 else 0.3)) * sway + 0.1 * rng.random()
                    out.append((s, min(1.0, vel)))
        elif slot == "ride":
            if section == "drop" or energy > 0.85:
                for s in range(0, S, 2):
                    if rng.random() < dens:
                        out.append((s, (0.85 if s % 4 == 0 else 0.55) + 0.1 * rng.random()))
        elif slot == "ohat":
            if kind == "halftime":
                if rng.random() < 0.4 + 0.4 * energy:
                    out.append((S // 2 + S // 8, 0.7))
            else:
                if rng.random() < 0.6 + 0.4 * energy:
                    out.append((S // 4 + S // 8, 0.8))     # the classic off-beat open hat
                if energy > 0.7 and rng.random() < 0.5:
                    out.append((3 * S // 4 + S // 8, 0.7))
        elif slot == "perc":
            if bar_in_phrase == 0 and rng.random() < 0.15:     # slow drift of the perc cell
                self._perc_rot = (self._perc_rot + rng.choice([-1, 1])) % S
                self._perc_len = 12 if (ctx.get("poly") or rng.random() < 0.3) else 16
            L = self._perc_len
            pat = euclid(min(self._perc_k, L), L, self._perc_rot % L)
            base = (ctx.get("bar", 0) * S) % L
            for s in range(S):
                if pat[(base + s) % L] and rng.random() < dens * (0.5 + 0.5 * energy):
                    out.append((s, 0.5 + 0.4 * rng.random()))
        return out
