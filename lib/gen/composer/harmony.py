"""Chord loops. A progression (list of scale degrees, one per bar) is
picked from the style's grammar and held for a few phrases, so the ear
gets a loop to hold on to; a new one is drawn at a phrase boundary."""
from __future__ import annotations

import random

from lib.gen.theory import Key

ROMAN = ["i", "ii", "III", "iv", "v", "VI", "VII"]
ROMAN_MAJ = ["I", "ii", "iii", "IV", "V", "vi", "vii"]


class Harmony:
    def __init__(self, style: dict, key: Key, rng: random.Random):
        self.style = style
        self.key = key
        self.rng = rng
        self.progression = list(rng.choice(style["progressions"]))
        self.hold = rng.randint(*style["progression_hold"])
        self.phrases_on = 0

    def next_phrase(self, nbars: int) -> list:
        """Per-bar (degree, label) for the next phrase; rotates the loop."""
        self.phrases_on += 1
        if self.phrases_on > self.hold:
            self.progression = list(self.rng.choice(self.style["progressions"]))
            self.hold = self.rng.randint(*self.style["progression_hold"])
            self.phrases_on = 1
        labels = ROMAN if self.key.mode != "major" else ROMAN_MAJ
        out = []
        for b in range(nbars):
            deg = self.progression[b % len(self.progression)]
            out.append((deg, labels[deg % 7]))
        return out

    def chord_notes(self, degree: int, octave: int, size: int = 3) -> list:
        return self.key.chord(degree, octave=octave, size=size)

    def modulate(self, new_key: Key):
        self.key = new_key
        self.phrases_on = self.hold  # force a fresh progression
