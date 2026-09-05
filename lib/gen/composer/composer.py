"""Composer: turns style + key + tempo + an arc into Phrases of NoteEvents,
four bars at a time, on an integer sample clock.

Deterministic under a seed. All musical decisions happen here; the synth
backends only render what they are told. Steering (energy bias, density,
key/tempo changes) takes effect at the next phrase boundary - the unit at
which a listener hears a change as intentional."""
from __future__ import annotations

import random

from lib.gen import RATE
from lib.gen.events import NoteEvent, Phrase, DRUM_SLOTS
from lib.gen.composer.form import Form
from lib.gen.composer.harmony import Harmony
from lib.gen.composer.melody import Melody
from lib.gen.composer.rhythm import Drums
from lib.gen.composer.styles import get_style
from lib.gen.theory import Key, parse_key

PHRASE_BARS = 4
BEATS_PER_BAR = 4


class Composer:
    def __init__(self, style: str = "groove", bpm: float | None = None,
                 key="8A", seed: int = 1, arc_fn=None):
        self.style_name = style
        self.style = get_style(style)
        self.rng = random.Random(seed)
        self.seed = seed
        lo, hi = self.style["bpm"]
        self.bpm = float(bpm) if bpm else self.rng.uniform(lo, hi)
        self.key = key if isinstance(key, Key) else parse_key(key)
        if self.key.mode == "minor" and self.style["mode"] != "minor" and not isinstance(key, Key):
            # a Camelot 'A' key names a tonal centre; the style picks the colour
            self.key = Key(self.key.root, self.style["mode"])
        self.form = Form(self.style, self.rng, arc_fn)
        self.harmony = Harmony(self.style, self.key, self.rng)
        self.melody = Melody(self.style, self.key, self.rng)
        self.drums = Drums(self.style, self.rng)
        self.clock = 0              # sample of the next phrase's downbeat
        self.bar = 0
        # live steering (applied at phrase boundaries)
        self.energy_bias = 0.0      # -0.5 .. +0.5
        self.density = 1.0          # scales probabilistic layers
        self.swing = self.style["swing"]
        self.muted = set()
        self._pending_key = None
        self._pending_bpm = None
        self.log = []
        # Optional external note source (a Strudel pattern): when set, its
        # events replace the rule-generated ones; form/harmony/energy keep
        # running and are handed to it as context.
        self.pattern_source = None

    # -- steering -----------------------------------------------------------
    def set_key(self, key):
        self._pending_key = key if isinstance(key, Key) else parse_key(key)

    def set_bpm(self, bpm: float):
        self._pending_bpm = float(bpm)

    def request_end(self):
        self.form.ending = True

    def set_arc(self, arc_fn):
        self.form.arc_fn = arc_fn or (lambda bar: 0.6)

    def reseed(self, seed: int):
        """New randomness from the next phrase on; motif memory survives
        (identity), the dice change."""
        self.seed = int(seed)
        self.rng.seed(self.seed)

    # -- timing helpers ------------------------------------------------------
    @property
    def samples_per_beat(self) -> float:
        return RATE * 60.0 / self.bpm

    @property
    def samples_per_bar(self) -> int:
        return int(round(self.samples_per_beat * BEATS_PER_BAR))

    def _step_to_sample(self, bar0_sample: int, step_abs: int) -> int:
        S = self.style["steps_per_bar"]
        bar, step = divmod(step_abs, S)
        spb = self.samples_per_bar
        sps = spb / S
        t = bar * spb + step * sps
        if self.swing > 0 and step % 2 == 1:        # delay the off-16ths
            t += self.swing * sps * 2.0 * 0.5
        return int(round(bar0_sample + t))

    # -- the phrase -----------------------------------------------------------
    def next_phrase(self) -> Phrase:
        if self._pending_key is not None:
            self.key = self._pending_key
            self.harmony.modulate(self.key)
            self.melody.key = self.key
            self._pending_key = None
        if self._pending_bpm is not None:
            self.bpm = self._pending_bpm
            self._pending_bpm = None
        nb = PHRASE_BARS
        S = self.style["steps_per_bar"]
        section = self.form.section
        energy = max(0.0, min(1.0, self.form.energy() + self.energy_bias))
        layers = self.form.layers() - self.muted
        chords = self.harmony.next_phrase(nb)
        start = self.clock
        spb = self.samples_per_bar
        sps = spb / S
        last_of_section = self.form.bars_left <= nb
        ev = []
        if self.pattern_source is not None:
            ctx = {"energy": round(energy, 3), "section": section, "bar": self.bar,
                   "key": self.key.camelot, "bpm": round(self.bpm, 2), "phrase": self.bar // nb,
                   "chords": [c[1] for c in chords]}
            try:
                ev = list(self.pattern_source.events(self.bar, nb, start, spb, ctx))
                self.pattern_source.error = ""
            except Exception as e:  # noqa: BLE001 - the pattern must never kill the show
                self.pattern_source.error = f"{type(e).__name__}: {e}"
                ev = []
            return self._finish_phrase(ev, nb, start, spb, section, energy, chords, layers, None)

        def note(step_abs, slot, pitch, vel, dur_steps, **params):
            at = self._step_to_sample(start, step_abs)
            dur = max(int(dur_steps * sps) - 1, int(0.02 * RATE))
            ev.append(NoteEvent(at, slot, float(pitch), float(min(1.0, vel)), dur, params))

        # drums
        for slot in DRUM_SLOTS:
            if slot not in layers:
                continue
            for b in range(nb):
                for step, vel in self.drums.bar(slot, energy * self.density if slot != "kick" else energy,
                                                b, nb, last_of_section):
                    if step < S:
                        note(b * S + step, slot, 36.0, vel, 1)
        # bass
        if "bass" in layers:
            for b in range(nb):
                for step, midi, vel, gate in self.melody.bass_bar(chords[b][0], energy, b):
                    note(b * S + step, "bass", midi, vel, gate)
        # pad: one voicing per chord change, held to the next change
        if "pad" in layers:
            prev = None
            for b in range(nb):
                if prev is not None and chords[b][0] == prev[0]:
                    prev = (prev[0], prev[1] + 1, prev[2], prev[3])
                    continue
                if prev is not None:
                    for m in prev[2]:
                        note(prev[3] * S, "pad", m, 0.6, prev[1] * S)
                prev = (chords[b][0], 1, self.melody.pad_bar(chords[b][0], energy), b)
            if prev is not None:
                for m in prev[2]:
                    note(prev[3] * S, "pad", m, 0.6, prev[1] * S, sustain=True)
        # keys stabs
        if "keys" in layers:
            for b in range(nb):
                for step, chord, vel, gate in self.melody.keys_bar(chords[b][0], energy * self.density):
                    for m in chord:
                        note(b * S + step, "keys", m, vel, gate)
        # arp
        if "arp" in layers:
            for b in range(nb):
                for step, midi, vel, gate in self.melody.arp_bar(chords[b][0], energy * self.density):
                    note(b * S + step, "arp", midi, vel, gate)
        # lead motif
        op = None
        if "lead" in layers:
            notes, op = self.melody.lead_phrase(chords, energy * self.density, nb)
            for step_abs, midi, vel, gate in notes:
                note(step_abs, "lead", midi, vel, gate)

        return self._finish_phrase(ev, nb, start, spb, section, energy, chords, layers, op)

    def _finish_phrase(self, ev, nb, start, spb, section, energy, chords, layers, op):
        ev.sort(key=lambda e: (e.at, e.slot))
        drops = []
        drop_bar = self.form.upcoming_drop_bar()
        if drop_bar is not None:
            drops.append(start + (drop_bar - self.bar) * spb)
        phrase = Phrase(bar0=self.bar, nbars=nb, start=start, bpm=self.bpm,
                        key=self.key.name, section=section, energy=energy,
                        chords=chords, events=ev,
                        meta={"end": start + nb * spb, "drops": drops,
                              "layers": sorted(layers), "lead_op": op,
                              "camelot": self.key.camelot})
        self.log.append((self.bar, section, round(energy, 2), [c[1] for c in chords], op))
        self.clock = start + nb * spb
        self.bar += nb
        self.form.advance(nb)
        return phrase

    def phrases_until(self, total_samples: int):
        """Yield phrases until the composed timeline covers total_samples."""
        while self.clock < total_samples:
            yield self.next_phrase()
