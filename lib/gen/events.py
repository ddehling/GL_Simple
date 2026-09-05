"""The note-event contract between composer, scheduler and every synth
backend. Times are integer SAMPLE positions on the rack's own clock at
44100 Hz - the same idiom as DJSubmix automation ("at" stamps)."""
from __future__ import annotations

from dataclasses import dataclass, field

# Instrument slots a style may populate. A style maps each slot to a
# patch (analog voice + params, or a SoundFont program, or an SC synthdef).
SLOTS = ("kick", "snare", "hat", "ohat", "perc", "bass", "lead", "pad", "arp", "keys")
DRUM_SLOTS = ("kick", "snare", "hat", "ohat", "perc")


@dataclass(frozen=True)
class NoteEvent:
    at: int              # start sample
    slot: str            # one of SLOTS
    pitch: float         # MIDI note (drums: nominal)
    vel: float           # 0..1
    dur: int             # gate length in samples (release follows)
    params: dict = field(default_factory=dict)   # per-note modulation

    @property
    def end(self) -> int:
        return self.at + self.dur


@dataclass
class Phrase:
    """A contiguous block of bars the composer has fully decided."""
    bar0: int
    nbars: int
    start: int           # sample of bar0's downbeat
    bpm: float
    key: str             # Key.name at the time
    section: str
    energy: float        # 0..1 target the phrase was written for
    chords: list         # per bar: (degree, label)
    events: list         # NoteEvent, sorted by .at
    meta: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        return self.meta.get("end", self.start)

    def drops(self) -> list:
        """Sample positions of engineered drops inside this phrase."""
        return list(self.meta.get("drops", []))
