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
        self.taste = {}            # {section: weight multiplier} from the preference memory
        self.script = None         # [entry dicts] when a SongScript drives the form (lib/gen/script.py)
        self.script_i = -1
        self.script_end = False    # after the script: outro + stop (else the grammar takes over)

    # -- scripted mode --------------------------------------------------------
    def set_script(self, entries, end=True):
        """Follow `entries` ([{section, bars, ...}]) from now: the current
        section is replaced by the first entry immediately."""
        self.script = [dict(e) for e in entries] or None
        self.script_end = bool(end)
        self.script_i = -1
        if self.script:
            self._enter_script(0)

    def _enter_script(self, i):
        e = self.script[i]
        self.script_i = i
        self.section = e["section"] if e["section"] in self.style["sections"] else self.section
        self.bars_left = max(4, int(e.get("bars", 8)))
        self.bars_in = 0
        self.history.append((self._bar, self.section, self.bars_left))

    @property
    def script_entry(self):
        if self.script and 0 <= self.script_i < len(self.script):
            return self.script[self.script_i]
        return None

    @property
    def script_next(self):
        if self.script and 0 <= self.script_i + 1 < len(self.script):
            return self.script[self.script_i + 1]
        return None

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
            weighted.append((name, max(0.05, w * bias * float(self.taste.get(name, 1.0)))))
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
        if self.bars_left <= 0 and self.script:
            if self.script_i + 1 < len(self.script):
                self._enter_script(self.script_i + 1)
                return
            self.script = None          # the script is over
            if self.script_end:
                self.ending = True
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
        e = self.script_entry
        if e is not None and e.get("energy") is not None:
            base = float(e["energy"])
            if self.section not in ("build", "outro"):
                return max(0.0, min(1.0, base))
        arc = float(self.arc_fn(self._bar))
        e = 0.65 * base + 0.35 * arc
        if self.section == "build":
            e = base * (0.85 + 0.15 * (self.bars_in / max(1, self.bars_in + self.bars_left)))
        elif self.section == "outro":
            e = base * max(0.2, 1.0 - self.bars_in / max(1, self.bars_in + self.bars_left))
        return max(0.0, min(1.0, e))

    def layers(self) -> set:
        e = self.script_entry
        if e is not None and e.get("layers"):
            return set(e["layers"]) & set(self.style["slots"].keys())
        lay = self.style["sections"][self.section]["layers"]
        if "*" in lay:
            return set(self.style["slots"].keys())
        return set(lay) & set(self.style["slots"].keys())

    def upcoming_drop_bar(self):
        """Bar index of the next section that starts with a drop, if the
        current section is a build (the visuals count down to it)."""
        if self.section == "build":
            return self._bar + self.bars_left
        nxt = self.script_next
        if nxt is not None and nxt.get("section") == "drop" and self.section != "drop":
            return self._bar + self.bars_left
        return None
