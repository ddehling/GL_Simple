"""Drum and rhythmic grids: 16 steps per bar, velocities 0..1.

Euclidean patterns give perc parts that sound 'placed' rather than random;
probability grids give hats their life; kicks and claps are idiomatic and
energy-gated so the section masks (form.py) do the big moves and energy
does the small ones."""
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


class Drums:
    def __init__(self, style: dict, rng: random.Random):
        self.style = style
        self.rng = rng
        self.steps = style["steps_per_bar"]
        self._perc_rot = rng.randrange(self.steps)
        self._perc_k = rng.choice([3, 5, 7])
        self._bar_in_phrase = 0

    def _fill(self, bar_in_phrase: int, nbars: int) -> bool:
        return bar_in_phrase == nbars - 1

    def bar(self, slot: str, energy: float, bar_in_phrase: int, nbars: int,
            last_of_section: bool) -> list:
        """Return [(step, vel), ...] for one bar of one drum slot."""
        S = self.steps
        rng = self.rng
        dens = self.style["density"].get(slot, 1.0)
        out = []
        fill = last_of_section and self._fill(bar_in_phrase, nbars)
        if slot == "kick":
            if self.style["bpm"][0] >= 110:            # four on the floor
                for q in range(4):
                    out.append((q * S // 4, 1.0))
                if energy > 0.75 and rng.random() < 0.25:
                    out.append((S - 2, 0.6))          # pickup before the bar
            else:                                      # broken: 1, and-of-2, 3-ish
                out.append((0, 1.0))
                out.append((S * 5 // 8, 0.85) if rng.random() < 0.7 else (S // 2, 0.9))
                if rng.random() < 0.35 * energy:
                    out.append((S * 7 // 8 + (0 if rng.random() < 0.5 else -1), 0.6))
            if fill:
                out = [(0, 1.0), (S // 4, 1.0)] + [(S // 2 + i * 2, 0.7 + 0.1 * i) for i in range(4)]
        elif slot == "snare":
            out = [(S // 4, 0.95), (3 * S // 4, 1.0)]
            if energy > 0.6 and rng.random() < 0.3:
                out.append((3 * S // 4 + 3, 0.45))    # ghost
            if fill:
                out += [(S // 2 + 2 * i + 1, 0.5 + 0.12 * i) for i in range(4)]
        elif slot == "hat":
            base = 2 if energy < 0.5 else 1          # 8ths below half energy, 16ths above
            for s in range(0, S, base):
                onbeat = (s % 4 == 0)
                offbeat = (s % 4 == 2)
                p = dens * (0.95 if offbeat else (0.8 if onbeat else 0.45 + 0.5 * energy))
                if rng.random() < p:
                    vel = 0.9 if offbeat else (0.55 if onbeat else 0.4 + 0.3 * rng.random())
                    out.append((s, vel))
        elif slot == "ohat":
            if rng.random() < 0.6 + 0.4 * energy:
                out.append((S // 4 + S // 8, 0.8))     # the classic off-beat open hat
            if energy > 0.7 and rng.random() < 0.5:
                out.append((3 * S // 4 + S // 8, 0.7))
        elif slot == "perc":
            if rng.random() < 0.15:                   # slow drift of the perc cell
                self._perc_rot = (self._perc_rot + rng.choice([-1, 1])) % S
            pat = euclid(self._perc_k, S, self._perc_rot)
            for s, hit in enumerate(pat):
                if hit and rng.random() < dens * (0.5 + 0.5 * energy):
                    out.append((s, 0.5 + 0.4 * rng.random()))
        return out
