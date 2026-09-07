"""Chord loops with colour. A progression (list of scale degrees, one per
bar) is picked from the style's grammar and held for a few phrases, so
the ear gets a loop to hold on to; a new one is drawn at a phrase
boundary.

On top of the loop, per bar:
  * borrowed chords - a chord from the parallel mode (major IV or V in a
    minor key, minor iv or v in a major key) on a non-downbeat bar;
  * suspensions - sus4 / sus2 on the bar before a chord change, so the
    change resolves;
  * pedal - in the sections a style names, the bass holds the tonic
    while the chords move above it;
  * slow harmonic rhythm - in the sections a style names, the chord
    changes every two bars.

A bar's chord is (degree, label, spec); spec = {"sus": 0|2|4,
"alt": {chord_tone_index: semitones}, "pedal": bool, "borrowed": bool}.
Consumers that only want the root keep using chord[0]; chord[1] is the
display label; notes() spells the chord with the colour applied."""
from __future__ import annotations

import random

from lib.gen.theory import Key, MINOR_LIKE

ROMAN = ["i", "ii", "III", "iv", "v", "VI", "VII"]
ROMAN_MAJ = ["I", "ii", "iii", "IV", "V", "vi", "vii"]
PLAIN = {"sus": 0, "alt": {}, "pedal": False, "borrowed": False}


class Harmony:
    def __init__(self, style: dict, key: Key, rng: random.Random):
        self.style = style
        self.key = key
        self.rng = rng
        self.cfg = dict({"borrow": 0.2, "sus": 0.2, "pedal_in": [], "slow_in": []}, **style.get("harmony", {}))
        self.progression = list(rng.choice(style["progressions"]))
        self.hold = rng.randint(*style["progression_hold"])
        self.phrases_on = 0
        self.override = None       # scripted degrees per bar (plain, no colour) while set

    def _borrow(self, degree: int):
        """Alterations that borrow this degree's chord from the parallel
        mode, or None when the degree has no idiomatic borrowing."""
        minor = self.key.mode in MINOR_LIKE
        if minor and degree in (3, 4):          # iv -> IV (dorian), v -> V (harmonic)
            return {1: +1}
        if not minor and degree in (3, 4):      # IV -> iv, V -> v (modal mixture)
            return {1: -1}
        if minor and degree == 6:               # VII -> vii dim-ish colour: flatten the 5th
            return {2: -1}
        return None

    def next_phrase(self, nbars: int, section: str | None = None) -> list:
        """Per-bar (degree, label, spec) for the next phrase; rotates the loop."""
        rng = self.rng
        self.phrases_on += 1
        if self.phrases_on > self.hold:
            self.progression = list(rng.choice(self.style["progressions"]))
            self.hold = rng.randint(*self.style["progression_hold"])
            self.phrases_on = 1
        labels = ROMAN if self.key.mode != "major" else ROMAN_MAJ
        if self.override:
            prog = list(self.override)
            return [self.scripted(prog[b % len(prog)], labels) for b in range(nbars)]
        slow = section in self.cfg["slow_in"]
        pedal = section in self.cfg["pedal_in"]
        degs = []
        for b in range(nbars):
            idx = (b // 2) if slow else b
            degs.append(self.progression[idx % len(self.progression)])
        out = []
        for b, deg in enumerate(degs):
            spec = {"sus": 0, "alt": {}, "pedal": pedal, "borrowed": False}
            label = labels[deg % 7]
            nxt = degs[b + 1] if b + 1 < nbars else None
            changes = nxt is not None and nxt != deg
            if b > 0 and rng.random() < self.cfg["borrow"]:
                alt = self._borrow(deg % 7)
                if alt:
                    spec["alt"] = alt
                    spec["borrowed"] = True
                    label = label.upper() if alt.get(1) == 1 else label.lower()
                    label += "*"
            if changes and not spec["borrowed"] and rng.random() < self.cfg["sus"]:
                spec["sus"] = 4 if rng.random() < 0.7 else 2
                label += f"sus{spec['sus']}"
            out.append((deg, label, spec))
        return out

    def scripted(self, entry, labels=None) -> tuple:
        """A scripted bar: a degree (int) or {"deg", "third": maj|min,
        "sus": 2|4} -> (degree, label, spec). A third that is not the
        key's own is an altered chord tone (the song borrowed it)."""
        labels = labels or (ROMAN if self.key.mode != "major" else ROMAN_MAJ)
        if isinstance(entry, dict):
            deg = int(entry.get("deg", 0)) % 7
            spec = dict(PLAIN, alt={})
            label = labels[deg]
            third = entry.get("third")
            if third in ("maj", "min"):
                own = (self.key.degree_pc(deg + 2) - self.key.degree_pc(deg)) % 12       # 3 = minor, 4 = major
                want = 4 if third == "maj" else 3
                if own != want:
                    spec["alt"] = {1: want - own}
                    spec["borrowed"] = True
                    label = (label.upper() if want == 4 else label.lower()) + "*"
            sus = int(entry.get("sus", 0) or 0)
            if sus in (2, 4):
                spec["sus"] = sus
                label += f"sus{sus}"
            return (deg, label, spec)
        deg = int(entry) % 7
        return (deg, labels[deg], dict(PLAIN, alt={}))

    def notes(self, chord, octave: int, size: int = 3, extra=()) -> list:
        """Spell a bar's chord (degree, label, spec) or a bare degree."""
        if isinstance(chord, (tuple, list)):
            deg, spec = chord[0], (chord[2] if len(chord) > 2 else PLAIN)
        else:
            deg, spec = int(chord), PLAIN
        notes = self.key.chord(deg, octave=octave, size=size, extra=extra)
        sus = spec.get("sus", 0)
        if sus and len(notes) > 1:
            notes[1] = self.key.degree_midi(deg + (3 if sus == 4 else 1), octave)
        for idx, semis in spec.get("alt", {}).items():
            if idx < len(notes):
                notes[idx] += semis
        return notes

    def root_degree(self, chord) -> int:
        """The bass degree: the tonic under a pedal, else the chord root."""
        if isinstance(chord, (tuple, list)) and len(chord) > 2 and chord[2].get("pedal"):
            return 0
        return chord[0] if isinstance(chord, (tuple, list)) else int(chord)

    def chord_notes(self, degree: int, octave: int, size: int = 3) -> list:
        return self.key.chord(degree, octave=octave, size=size)

    def modulate(self, new_key: Key):
        self.key = new_key
        self.phrases_on = self.hold  # force a fresh progression
