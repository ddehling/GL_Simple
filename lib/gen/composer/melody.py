"""Pitched parts: bass, lead (with motif memory), pads, arps, keys.

Motif memory is what makes an hour hang together: a lead phrase is stored
as (degree offsets, rhythm); later phrases REPEAT, TRANSPOSE (follow the
chord), VARY (swap a few notes), INVERT or RETIRE it. Bounded (LRU) so
memory stays flat over a night."""
from __future__ import annotations

import random
from collections import OrderedDict

from lib.gen.theory import Key


class Motif:
    __slots__ = ("steps", "degrees", "uses", "born")

    def __init__(self, steps, degrees, born):
        self.steps = list(steps)        # step indices within the phrase grid
        self.degrees = list(degrees)    # scale-degree offsets from chord root
        self.uses = 0
        self.born = born


class MotifMemory:
    def __init__(self, rng: random.Random, capacity: int = 12):
        self.rng = rng
        self.cap = capacity
        self._m = OrderedDict()
        self._n = 0

    def add(self, motif: Motif):
        self._n += 1
        self._m[self._n] = motif
        if len(self._m) > self.cap:
            self._m.popitem(last=False)
        return self._n

    def pick(self):
        if not self._m:
            return None
        # prefer recent and not-yet-worn-out motifs
        keys = list(self._m)
        weights = [1.0 / (1.0 + m.uses * 0.5) * (1.0 + 0.1 * i) for i, m in enumerate(self._m.values())]
        tot = sum(weights)
        r = self.rng.random() * tot
        for k, w in zip(keys, weights):
            r -= w
            if r <= 0:
                self._m[k].uses += 1
                self._m.move_to_end(k)
                return self._m[k]
        m = self._m[keys[-1]]
        m.uses += 1
        return m

    def __len__(self):
        return len(self._m)


class Melody:
    def __init__(self, style: dict, key: Key, rng: random.Random):
        self.style = style
        self.key = key
        self.rng = rng
        self.steps = style["steps_per_bar"]
        self.memory = MotifMemory(rng)
        self._bass_cell = None
        self._bass_cell_left = 0
        self._phrase_count = 0

    # -- bass -------------------------------------------------------------
    def bass_bar(self, degree: int, energy: float, bar_in_phrase: int) -> list:
        """[(step, midi, vel, dur_steps)] for one bar. A bass CELL (rhythm +
        contour) persists for a few bars so it reads as a riff."""
        S = self.steps
        rng = self.rng
        dens = self.style["density"].get("bass", 0.7)
        if self._bass_cell is None or self._bass_cell_left <= 0:
            n_hits = max(2, int(round((3 + 6 * energy) * dens)))
            from lib.gen.composer.rhythm import euclid
            pat = euclid(min(n_hits, S), S, 0)
            contour = []
            for s, hit in enumerate(pat):
                if not hit:
                    continue
                r = rng.random()
                off = 0 if r < 0.55 else (7 if r < 0.7 else (4 if r < 0.85 else -2))
                octave_up = 12 if (rng.random() < 0.2 * energy) else 0
                contour.append((s, off + octave_up))
            self._bass_cell = contour
            self._bass_cell_left = rng.randint(2, 4)
        self._bass_cell_left -= 1
        oct_ = self.style["slots"]["bass"].get("octave", 1)
        root = self.key.degree_midi(degree, oct_)
        out = []
        cell = self._bass_cell
        for i, (s, off) in enumerate(cell):
            nxt = cell[i + 1][0] if i + 1 < len(cell) else S
            gate = max(1, int((nxt - s) * (0.5 + 0.4 * (1 - energy))))
            pitch = self.key.snap(root + off) if off not in (0, 12) else root + off
            vel = 0.85 if s % 4 == 0 else 0.7
            out.append((s, pitch, vel, gate))
        return out

    # -- lead -------------------------------------------------------------
    def lead_phrase(self, chords: list, energy: float, nbars: int) -> list:
        """[(step_abs, midi, vel, dur_steps)] over the phrase, using or
        creating a motif. step_abs counts from the phrase start."""
        S = self.steps
        rng = self.rng
        self._phrase_count += 1
        dens = self.style["density"].get("lead", 0.4)
        motif = None
        op = "new"
        if len(self.memory) and rng.random() < 0.7:
            motif = self.memory.pick()
            op = rng.choice(["repeat", "repeat", "transpose", "vary", "invert"])
        if motif is None:
            n = max(3, int(round(4 + 8 * energy * dens)))
            steps = sorted(rng.sample(range(0, 2 * S, 2 if energy > 0.5 else 4), min(n, S)))
            degs, cur = [], 0
            for _ in steps:
                cur += rng.choice([-2, -1, -1, 0, 1, 1, 2, 3])
                cur = max(-4, min(9, cur))
                degs.append(cur)
            motif = Motif(steps, degs, self._phrase_count)
            self.memory.add(motif)
        steps, degs = list(motif.steps), list(motif.degrees)
        if op == "vary":
            for i in range(len(degs)):
                if rng.random() < 0.3:
                    degs[i] += rng.choice([-1, 1])
        elif op == "invert":
            degs = [-d for d in degs]
        oct_ = self.style["slots"]["lead"].get("octave", 4)
        out = []
        # motif spans 2 bars; place it on bars 0-1 and (varied) 2-3
        for rep in range(0, nbars, 2):
            for s, d in zip(steps, degs):
                bar = rep + s // S
                if bar >= nbars:
                    break
                deg_root = chords[bar][0] if op == "transpose" or rep == 0 else chords[0][0]
                midi = self.key.degree_midi(deg_root + d, oct_)
                if rep and rng.random() < 0.15:      # a little answer-phrase life
                    continue
                vel = 0.6 + 0.3 * rng.random()
                out.append((rep * S + s, midi, vel, rng.choice([1, 2, 2, 3, 4])))
        return out, op

    # -- pad / keys / arp --------------------------------------------------
    def pad_bar(self, degree: int, energy: float) -> list:
        oct_ = self.style["slots"]["pad"].get("octave", 3)
        size = 4 if energy > 0.5 else 3
        notes = self.key.chord(degree, octave=oct_, size=size)
        if self.rng.random() < 0.5:
            notes = self._voice_lead(notes)
        return notes

    def _voice_lead(self, notes):
        # drop the 3rd an octave for a wider voicing now and then
        return [notes[0]] + [notes[1] - 12] + notes[2:]

    def keys_bar(self, degree: int, energy: float) -> list:
        """Stabs: [(step, [midis], vel, dur_steps)]."""
        S = self.steps
        rng = self.rng
        dens = self.style["density"].get("keys", 0.35)
        oct_ = self.style["slots"]["keys"].get("octave", 3)
        chord = self.key.chord(degree, octave=oct_, size=3, extra=(6,) if rng.random() < 0.4 else ())
        out = []
        cands = [2, 6, 10, 14, 7, 11] if energy > 0.45 else [0, 8]
        for s in cands:
            if rng.random() < dens * (0.5 + 0.5 * energy):
                out.append((s, chord, 0.55 + 0.3 * rng.random(), rng.choice([1, 2, 3])))
        return out

    def arp_bar(self, degree: int, energy: float) -> list:
        """[(step, midi, vel, dur_steps)] 16th arpeggio over the chord."""
        S = self.steps
        rng = self.rng
        oct_ = self.style["slots"]["arp"].get("octave", 4)
        chord = self.key.chord(degree, octave=oct_, size=4)
        seq = chord + [chord[0] + 12]
        if rng.random() < 0.5:
            seq = seq[::-1]
        step = 1 if energy > 0.6 else 2
        dens = self.style["density"].get("arp", 0.8)
        out = []
        for i, s in enumerate(range(0, S, step)):
            if rng.random() < dens:
                out.append((s, seq[i % len(seq)], 0.45 + 0.3 * (1 if s % 4 == 0 else rng.random()), 1))
        return out
