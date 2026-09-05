"""Keys, scales, Camelot wheel and chord spelling for the composer.

Pitch classes are 0..11 with C = 0. MIDI note = 12 * (octave + 1) + pc.
"""
from __future__ import annotations

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_PC = {n: i for i, n in enumerate(NOTE_NAMES)}
_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

# Interval patterns (semitones from the root), 7 degrees each.
SCALES = {
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "minor":      [0, 2, 3, 5, 7, 8, 10],      # aeolian
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "phrygian":   [0, 1, 3, 5, 7, 8, 10],
    "lydian":     [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "harmonic":   [0, 2, 3, 5, 7, 8, 11],
}
MINOR_LIKE = {"minor", "dorian", "phrygian", "harmonic"}

# Camelot wheel: number -> (minor root pc for "A", major root pc for "B").
_CAMELOT_A = {1: 8, 2: 3, 3: 10, 4: 5, 5: 0, 6: 7, 7: 2, 8: 9, 9: 4, 10: 11, 11: 6, 12: 1}
_CAMELOT_B = {k: (v + 3) % 12 for k, v in _CAMELOT_A.items()}


class Key:
    """A tonal centre: root pitch class + scale name."""

    def __init__(self, root_pc: int, mode: str = "minor"):
        if mode not in SCALES:
            raise ValueError(f"unknown mode {mode!r}")
        self.root = int(root_pc) % 12
        self.mode = mode

    # -- naming ---------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{NOTE_NAMES[self.root]} {self.mode}"

    @property
    def camelot(self) -> str:
        minor_root = self.root if self.mode in MINOR_LIKE else (self.root - 3) % 12
        for num, pc in _CAMELOT_A.items():
            if pc == minor_root:
                return f"{num}{'A' if self.mode in MINOR_LIKE else 'B'}"
        return "?"

    def __repr__(self):
        return f"Key({self.name}, {self.camelot})"

    # -- pitches --------------------------------------------------------
    @property
    def intervals(self):
        return SCALES[self.mode]

    def degree_pc(self, degree: int) -> int:
        """Scale degree (0-based, any int; wraps) -> pitch class."""
        return (self.root + self.intervals[degree % 7]) % 12

    def degree_midi(self, degree: int, octave: int = 3) -> int:
        """Degree with octave carry: degree 7 == degree 0 an octave up."""
        octave += degree // 7
        return 12 * (octave + 1) + self.degree_pc(degree)

    def snap(self, midi: int) -> int:
        """Nearest scale tone to an arbitrary MIDI note (ties go down)."""
        pcs = {self.degree_pc(d) for d in range(7)}
        for delta in (0, -1, 1, -2, 2, -3, 3):
            if (midi + delta) % 12 in pcs:
                return midi + delta
        return midi

    def chord(self, degree: int, octave: int = 3, size: int = 3, extra=()) -> list:
        """Stacked-thirds chord on a scale degree, in-scale (diatonic).
        size 3 = triad, 4 = seventh; extra = extra degree offsets (e.g. 9)."""
        notes = [self.degree_midi(degree + 2 * i, octave) for i in range(size)]
        for e in extra:
            notes.append(self.degree_midi(degree + e, octave))
        return notes

    def relative(self, semitones: int, mode: str | None = None) -> "Key":
        return Key((self.root + semitones) % 12, mode or self.mode)

    def neighbours(self):
        """Camelot-adjacent keys (same energy neighbourhood): +/-1 on the
        wheel (fifths) and the relative major/minor."""
        out = [self.relative(7), self.relative(-7)]
        if self.mode in MINOR_LIKE:
            out.append(Key((self.root + 3) % 12, "major"))
        else:
            out.append(Key((self.root - 3) % 12, "minor"))
        return out


def parse_key(text: str) -> Key:
    """'8A' | '11B' (Camelot) | 'Am' | 'F#m' | 'C' | 'D dorian'."""
    t = text.strip()
    if len(t) >= 2 and t[-1] in "AB" and t[:-1].isdigit():
        num = int(t[:-1])
        if num not in _CAMELOT_A:
            raise ValueError(f"bad camelot {text!r}")
        return Key(_CAMELOT_A[num], "minor") if t[-1] == "A" else Key(_CAMELOT_B[num], "major")
    parts = t.split()
    name = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else None
    if mode is None:
        if name.endswith("m") and name[:-1] in _PC:
            return Key(_PC[name[:-1]], "minor")
        mode = "major"
    if name not in _PC:
        raise ValueError(f"bad key {text!r}")
    return Key(_PC[name], mode)


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def midi_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
