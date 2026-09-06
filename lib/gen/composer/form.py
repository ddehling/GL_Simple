"""Section state machine: which section we are in, for how many bars,
and the energy target it implies. Driven by the style's form grammar and
an external arc (0..1 over the set) that biases the grammar toward calm
or heat and bounds the section's energy."""
from __future__ import annotations

import random


class Form:
    def __init__(self, style: dict, rng: random.Random, arc_fn=None):
        self.style = style
        self.rng = rng
        self.arc_fn = arc_fn or (lambda bar: 0.6)
        self.section = style["first"]
        self.bars_left = self._draw_bars(self.section)
        self.bars_in = 0
        self.history = []          # (bar0, section, nbars)
        self._bar = 0
        self.history.append((0, self.section, self.bars_left))
        self.ending = False
        self.hold = False          # operator: stay in this section
        self.requested = None      # operator/director: go here next

    def _draw_bars(self, section: str) -> int:
        lo, hi = self.style["sections"][section]["bars"]
        # phrase discipline: multiples of 4 bars
        n = self.rng.randint(lo // 4, hi // 4) * 4
        return max(4, n)

    def request(self, section: str):
        """Make `section` the next section (at the current one's end; a
        request while >8 bars remain also shortens the wait to one phrase)."""
        if section in self.style["sections"]:
            self.requested = section
            if self.bars_left > 8:
                self.bars_left = 4

    def _next_section(self) -> str:
        if self.ending:
            return "outro"
        if self.requested:
            r, self.requested = self.requested, None
            return r
        arc = float(self.arc_fn(self._bar))
        opts = self.style["form"][self.section]
        weighted = []
        for name, w in opts:
            e = self.style["sections"][name]["energy"]
            # bias: hot arc favours hot sections, calm arc favours calm ones
            bias = 1.0 + 1.5 * (arc - 0.5) * (e - 0.5) * 2.0
            weighted.append((name, max(0.05, w * bias)))
        tot = sum(w for _, w in weighted)
        r = self.rng.random() * tot
        for name, w in weighted:
            r -= w
            if r <= 0:
                return name
        return weighted[-1][0]

    def advance(self, nbars: int):
        """Consume nbars; may cross into a new section (sections are
        multiples of 4 bars and phrases are 4 bars, so never mid-phrase)."""
        self._bar += nbars
        self.bars_in += nbars
        self.bars_left -= nbars
        if self.bars_left <= 0 and self.hold and not self.ending:
            self.bars_left = 4          # ride the section another phrase
            return
        if self.bars_left <= 0:
            self.section = self._next_section()
            self.bars_left = self._draw_bars(self.section)
            self.bars_in = 0
            self.history.append((self._bar, self.section, self.bars_left))

    def energy(self) -> float:
        """Energy target for the current bar: the section's own level,
        pulled toward the arc, with a ramp inside builds/outros."""
        sec = self.style["sections"][self.section]
        base = sec["energy"]
        arc = float(self.arc_fn(self._bar))
        e = 0.65 * base + 0.35 * arc
        if self.section == "build":
            e = base * (0.85 + 0.15 * (self.bars_in / max(1, self.bars_in + self.bars_left)))
        elif self.section == "outro":
            e = base * max(0.2, 1.0 - self.bars_in / max(1, self.bars_in + self.bars_left))
        return max(0.0, min(1.0, e))

    def layers(self) -> set:
        lay = self.style["sections"][self.section]["layers"]
        if "*" in lay:
            return set(self.style["slots"].keys())
        return set(lay) & set(self.style["slots"].keys())

    def upcoming_drop_bar(self):
        """Bar index of the next section that starts with a drop, if the
        current section is a build (the visuals count down to it)."""
        if self.section == "build":
            return self._bar + self.bars_left
        return None
