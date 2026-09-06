"""Pitched parts: bass, lead (with motif memory and a THEME), pads, arps,
keys.

Motif memory is what makes an hour hang together: a lead phrase is stored
as (degree offsets, rhythm, contour); later phrases develop it: REPEAT,
TRANSPOSE (follow the chord), VARY (swap a few notes), INVERT, SEQUENCE
(the cell again a step up or down), FRAGMENT (its head twice), AUGMENT
(twice as slow), RETROGRADE. Bounded (LRU) so memory stays flat over a
night. A thumbs-up from the operator boosts the motif that played.

Where motifs come from, best first: an AUTHORED hook (an LLM wrote it for
this style and key - lib/gen/composer/hooks.py - with an answer phrase),
else the CORPUS model (rhythm cells, an interval model and cadences from
public-domain melodies - melody_model.py), else the old random walk.

The theme is the motif a movement is about: made (or chosen) in the first
build, restated verbatim on the downbeat of the drop, retired when the
key moves. That is the difference between a loop and a track.

Phrase shape: bars 0-1 ask (the motif), bars 2-3 answer (the authored
answer, or the motif developed toward a cadence); the last phrase of a
section cadences on the tonic; the last phrase of a build climaxes an
octave up. Harmonic function underneath: strong beats are chord tones,
the note before a strong beat approaches by step, everything else is a
scale tone; a contour shapes the line. Pad voicings are voice-led.

The parts listen to each other: the bass avoids the kick's steps except
the downbeat (and slides between its own notes), the keys answer the
lead in the gaps it leaves, and every part accents by metric position."""
from __future__ import annotations

import math
import random
from collections import OrderedDict

from lib.gen.composer import melody_model
from lib.gen.theory import Key

CONTOURS = {
    "flat": lambda u: 0.0,
    "arch": lambda u: 3.0 * math.sin(math.pi * u),
    "rise": lambda u: 3.0 * u,
    "fall": lambda u: 3.0 * (1.0 - u),
    "wave": lambda u: 2.0 * math.sin(2.0 * math.pi * u),
}
DEVELOP_OPS = ("repeat", "repeat", "transpose", "vary", "invert", "sequence", "fragment", "augment", "retrograde")


class Motif:
    __slots__ = ("steps", "degrees", "uses", "born", "contour", "liked", "answer", "name")

    def __init__(self, steps, degrees, born, contour="flat", answer=None, name=""):
        self.steps = list(steps)        # step indices within the phrase grid (0..31: two bars)
        self.degrees = list(degrees)    # scale-degree offsets from chord root
        self.uses = 0
        self.born = born
        self.contour = contour
        self.liked = 0
        self.answer = answer            # optional (steps, degrees) for bars 2-3
        self.name = name


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
        # prefer recent, not-yet-worn-out and liked motifs
        keys = list(self._m)
        weights = [1.0 / (1.0 + m.uses * 0.5) * (1.0 + 0.1 * i) * (1.0 + 0.6 * m.liked)
                   for i, m in enumerate(self._m.values())]
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

    def boost(self, motif):
        if motif is not None:
            motif.liked += 1
            motif.uses = max(0, motif.uses - 2)

    def __len__(self):
        return len(self._m)


def _root_of(chord):
    return chord[0] if isinstance(chord, (tuple, list)) else int(chord)


def develop(op: str, steps, degs, rng: random.Random, S: int = 16):
    """Apply a development operator to a two-bar (steps, degrees) cell."""
    steps, degs = list(steps), list(degs)
    if op == "vary":
        degs = [d + (rng.choice([-1, 1]) if rng.random() < 0.3 else 0) for d in degs]
    elif op == "invert":
        degs = [-d for d in degs]
    elif op == "sequence":
        shift = rng.choice([-2, -1, 1, 2])
        half = [(s, d) for s, d in zip(steps, degs) if s < S]
        if len(half) >= 2:
            steps = [s for s, _ in half] + [s + S for s, _ in half]
            degs = [d for _, d in half] + [d + shift for _, d in half]
    elif op == "fragment":
        head = [(s, d) for s, d in zip(steps, degs) if s < S // 2 + 1]
        if len(head) >= 2:
            steps = [s for s, _ in head] + [s + S // 2 for s, _ in head] + [s + S for s, _ in head] + [s + S + S // 2 for s, _ in head]
            degs = [d for _, d in head] * 4
            keep = [i for i, s in enumerate(steps) if s < 2 * S]
            steps, degs = [steps[i] for i in keep], [degs[i] for i in keep]
    elif op == "augment":
        pairs = [(s * 2, d) for s, d in zip(steps, degs) if s * 2 < 2 * S]
        if len(pairs) >= 2:
            steps, degs = [s for s, _ in pairs], [d for _, d in pairs]
    elif op == "retrograde":
        degs = degs[::-1]
    return steps, degs


class Melody:
    def __init__(self, style: dict, key: Key, rng: random.Random, harmony=None):
        self.style = style
        self.key = key
        self.rng = rng
        self.harmony = harmony
        self.steps = style["steps_per_bar"]
        self.memory = MotifMemory(rng)
        self.theme = None
        self.last_motif = None
        self.hook_provider = None       # callable(rng) -> hook dict or None (lib/gen/composer/hooks.py)
        self.bass_override = None       # {"steps": [...], "degrees": [...]} from a SongScript (the source's bass line)
        self.source = "walk"            # where the last new motif came from: hook | corpus | walk
        self._bass_cell = None
        self._bass_cell_left = 0
        self._phrase_count = 0
        self._pad_prev = None

    def _notes(self, chord, octave, size=3, extra=()):
        if self.harmony is not None:
            return self.harmony.notes(chord, octave, size, extra)
        return self.key.chord(_root_of(chord), octave=octave, size=size, extra=extra)

    def _bass_root_degree(self, chord):
        if self.harmony is not None:
            return self.harmony.root_degree(chord)
        return _root_of(chord)

    def like(self):
        """Operator liked what is playing: favour the motif that played."""
        self.memory.boost(self.last_motif)

    # -- bass -------------------------------------------------------------
    def bass_bar(self, chord, energy: float, bar_in_phrase: int, kick_steps=()) -> list:
        """[(step, midi, vel, dur_steps)] for one bar. A bass CELL (rhythm +
        contour) persists for a few bars so it reads as a riff. Hits that
        would sit on a kick (other than the downbeat) move to the next free
        16th, so bass and kick interlock instead of masking each other."""
        S = self.steps
        rng = self.rng
        dens = self.style["density"].get("bass", 0.7)
        if self.bass_override and self.bass_override.get("steps"):
            oct_ = self.style["slots"]["bass"].get("octave", 1)
            tonic = self.key.degree_midi(0, oct_)
            out = []
            steps = [int(x) for x in self.bass_override["steps"]]
            degs = [int(x) for x in self.bass_override["degrees"]]
            for i, (st, d) in enumerate(zip(steps, degs)):
                if not (0 <= st < S):
                    continue
                nxt = steps[i + 1] if i + 1 < len(steps) else S
                gate = max(0.5, (nxt - st) * 0.8)
                out.append((st, self.key.degree_midi(d, oct_) if d >= -7 else tonic, (0.9 if st % 4 == 0 else 0.72), gate))
            return out
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
        root = self.key.degree_midi(self._bass_root_degree(chord), oct_)
        kicks = set(kick_steps)
        cell = []
        used = set()
        for s, off in self._bass_cell:
            if s != 0 and s in kicks:
                for cand in (s + 1, s + 2, s - 1):
                    if 0 <= cand < S and cand not in kicks and cand not in used:
                        s = cand
                        break
                else:
                    continue
            if s in used:
                continue
            used.add(s)
            cell.append((s, off))
        cell.sort()
        out = []
        for i, (s, off) in enumerate(cell):
            nxt = cell[i + 1][0] if i + 1 < len(cell) else S
            gate = max(0.5, (nxt - s) * (0.45 + 0.45 * (1 - energy)) * (0.85 + 0.3 * rng.random()))
            pitch = self.key.snap(root + off) if off not in (0, 12) else root + off
            vel = (0.9 if s % 4 == 0 else 0.7) + 0.08 * (rng.random() - 0.5)
            out.append((s, pitch, vel, gate))
        return out

    # -- lead: where motifs come from -----------------------------------
    def _walk_motif(self, energy, dens):
        S = self.steps
        rng = self.rng
        n = max(3, int(round(4 + 8 * energy * dens)))
        grid = list(range(0, 2 * S, 2 if energy > 0.5 else 4))
        w = [4.0 if g % 4 == 0 else (2.0 if g % 2 == 0 else 1.0) for g in grid]
        steps = []
        while grid and len(steps) < min(n, S):
            i = rng.choices(range(len(grid)), weights=w)[0]
            steps.append(grid.pop(i))
            w.pop(i)
        steps.sort()
        degs, cur = [], 0
        for _ in steps:
            cur += rng.choice([-2, -1, -1, 0, 1, 1, 2, 3])
            cur = max(-4, min(9, cur))
            degs.append(cur)
        return steps, degs, rng.choice(list(CONTOURS))

    def _new_motif(self, energy, dens, want_hook=False):
        """A fresh motif: an authored hook when asked for one and one is
        available, else the corpus model, else the random walk."""
        rng = self.rng
        n = max(3, int(round(4 + 8 * energy * dens)))
        answer = None
        name = ""
        if want_hook and self.hook_provider is not None:
            h = self.hook_provider(rng)
            if h:
                steps, degs, contour = list(h["steps"]), list(h["degrees"]), h.get("contour", "flat")
                if h.get("answer"):
                    answer = (list(h["answer"]["steps"]), list(h["answer"]["degrees"]))
                name = h.get("name", "")
                self.source = "hook"
                m = Motif(steps, degs, self._phrase_count, contour=contour, answer=answer, name=name)
                self.memory.add(m)
                return m
        steps = melody_model.sample_rhythm(rng, n, self.steps) if melody_model.available() else None
        if steps:
            degs = melody_model.sample_line(rng, steps, cadence=True)
            contour = melody_model.sample_contour(rng)
            self.source = "corpus"
        else:
            steps, degs, contour = self._walk_motif(energy, dens)
            self.source = "walk"
        m = Motif(steps, degs, self._phrase_count, contour=contour)
        self.memory.add(m)
        return m

    def _nearest_pc(self, midi, pcs):
        """Nearest pitch to midi whose pitch class is in pcs."""
        best = midi
        bd = 99
        for d in range(-6, 7):
            if (midi + d) % 12 in pcs and abs(d) < bd:
                best, bd = midi + d, abs(d)
        return best

    def _fit(self, s, next_s, chord, next_chord, raw, oct_):
        """Harmonic function for one lead note at step s (within the bar)."""
        chord_pcs = {m % 12 for m in self._notes(chord, oct_, 4)}
        if s % 4 == 0:                                   # strong beat: a chord tone
            return self._nearest_pc(raw, chord_pcs)
        if next_s is not None and next_s % 4 == 0 and 1 <= next_s - s <= 2:
            target_pcs = {m % 12 for m in self._notes(next_chord, oct_, 4)}
            target = self._nearest_pc(raw, target_pcs)
            # approach by scale step from the side the line is coming from
            below = self.key.snap(target - 1) if self.key.snap(target - 1) != target else self.key.snap(target - 2)
            above = self.key.snap(target + 1) if self.key.snap(target + 1) != target else self.key.snap(target + 2)
            return below if raw <= target else above
        return self.key.snap(raw)

    def lead_phrase(self, chords: list, energy: float, nbars: int, theme: str | None = None, ctx: dict | None = None) -> tuple:
        """[(step_abs, midi, vel, dur_steps)] over the phrase, using or
        creating a motif. theme: "make" = this phrase creates (or picks)
        the movement's theme; "state" = restate the theme verbatim on the
        chord roots (the drop); None = memory as usual. ctx: section,
        last_of_section (cadence), climax (register up)."""
        S = self.steps
        rng = self.rng
        ctx = ctx or {}
        self._phrase_count += 1
        dens = self.style["density"].get("lead", 0.4)
        motif = None
        op = "new"
        if theme == "state" and self.theme is not None:
            motif, op = self.theme, "theme"
        elif theme == "make":
            if self.theme is None:
                self.theme = self._new_motif(energy, dens, want_hook=True)
            motif, op = self.theme, "theme_make"
        elif len(self.memory) and rng.random() < 0.7:
            motif = self.memory.pick()
            op = rng.choice(DEVELOP_OPS)
        if motif is None:
            motif = self._new_motif(energy, dens)
        self.last_motif = motif
        steps, degs = list(motif.steps), list(motif.degrees)
        if op in ("vary", "invert", "sequence", "fragment", "augment", "retrograde"):
            steps, degs = develop(op, steps, degs, rng, S)
        contour = CONTOURS.get(motif.contour, CONTOURS["flat"])
        oct_ = self.style["slots"]["lead"].get("octave", 4)
        if ctx.get("climax"):
            oct_ += 1
        cadence = bool(ctx.get("last_of_section"))
        raw = []
        # bars 0-1: the motif (the question); bars 2-3: the answer
        cells = [(0, steps, degs, False)]
        if nbars >= 4:
            if motif.answer and op in ("theme", "theme_make", "repeat"):
                cells.append((2 * S, list(motif.answer[0]), list(motif.answer[1]), True))
            else:
                a_steps, a_degs = develop(rng.choice(("sequence", "vary", "retrograde", "repeat")), steps, degs, rng, S)
                cells.append((2 * S, a_steps, a_degs, True))
        for base, c_steps, c_degs, is_answer in cells:
            for s, d in zip(c_steps, c_degs):
                sa = base + s
                bar = sa // S
                if bar >= nbars:
                    break
                follow = op in ("transpose", "theme", "theme_make", "sequence") or base == 0
                deg_root = _root_of(chords[bar]) if follow else _root_of(chords[0])
                u = sa / float(nbars * S)
                midi = self.key.degree_midi(deg_root + d + int(round(contour(u))), oct_)
                if is_answer and op not in ("theme",) and rng.random() < 0.12:   # a little answer-phrase life
                    continue
                vel = (0.75 if s % 4 == 0 else 0.6) + 0.2 * rng.random()
                if op == "theme":
                    vel = min(1.0, vel + 0.1)        # the hook is played with conviction
                raw.append((sa, midi, vel, rng.choice([1, 2, 2, 3, 4])))
        raw.sort(key=lambda x: x[0])
        # harmonic function + articulation
        shaped = []
        for i, (sa, midi, vel, d) in enumerate(raw):
            bar = min(nbars - 1, sa // S)
            nxt = raw[i + 1][0] if i + 1 < len(raw) else None
            next_bar = min(nbars - 1, nxt // S) if nxt is not None else bar
            midi = self._fit(sa % S, (nxt - bar * S) if (nxt is not None and next_bar == bar) else None,
                             chords[bar], chords[next_bar], midi, oct_)
            gap = (nxt - sa) if nxt is not None else d
            d = min(d, gap) * (1.03 if rng.random() < 0.45 else rng.uniform(0.55, 0.9))
            shaped.append((sa, midi, vel, max(0.4, d)))
        if cadence and shaped:
            # the section's last phrase ends on the tonic, held
            sa, midi, vel, d = shaped[-1]
            tonic = self._nearest_pc(midi, {self.key.degree_pc(0)})
            shaped[-1] = (sa, tonic, min(1.0, vel + 0.05), max(d, 3.0))
        return shaped, op

    def retire_theme(self):
        self.theme = None

    # -- pad / keys / arp --------------------------------------------------
    def pad_bar(self, chord, energy: float) -> list:
        """Voice-led pad voicing: among the inversions of this chord, the
        one that moves least from the previous voicing."""
        oct_ = self.style["slots"]["pad"].get("octave", 3)
        size = 4 if energy > 0.5 else 3
        base = self._notes(chord, oct_, size)
        cands = []
        for k in range(len(base)):
            inv = base[k:] + [m + 12 for m in base[:k]]
            cands.append(inv)
            cands.append([m - 12 for m in inv])
        lo, hi = 12 * (oct_ + 1) - 7, 12 * (oct_ + 3)
        cands = [c for c in cands if min(c) >= lo and max(c) <= hi] or [base]
        if self._pad_prev is None:
            pick = base if self.rng.random() < 0.5 else self._voice_lead(base)
        else:
            prev = sorted(self._pad_prev)

            def cost(v):
                vs = sorted(v)
                return sum(abs(a - b) for a, b in zip(vs, prev)) + 2 * abs(len(vs) - len(prev))
            pick = min(cands, key=cost)
        self._pad_prev = list(pick)
        return pick

    def _voice_lead(self, notes):
        # drop the 3rd an octave for a wider voicing now and then
        return [notes[0]] + [notes[1] - 12] + notes[2:]

    def keys_bar(self, chord, energy: float, lead_steps=()) -> list:
        """Stabs: [(step, [midis], vel, dur_steps)]. With a lead playing,
        the keys ANSWER: they avoid the steps the lead occupies (+-1) and
        play at most two stabs, in its gaps."""
        S = self.steps
        rng = self.rng
        dens = self.style["density"].get("keys", 0.35)
        oct_ = self.style["slots"]["keys"].get("octave", 3)
        chord_notes = self._notes(chord, oct_, 3, extra=(6,) if rng.random() < 0.4 else ())
        if rng.random() < 0.4:                       # first inversion now and then
            chord_notes = chord_notes[1:] + [chord_notes[0] + 12]
        out = []
        cands = [2, 6, 10, 14, 7, 11] if energy > 0.45 else [0, 8]
        busy = set()
        for ls in lead_steps:
            busy.update((ls - 1, ls, ls + 1))
        answering = bool(lead_steps)
        for s in cands:
            if s in busy:
                continue
            if answering and len(out) >= 2:
                break
            if rng.random() < dens * (0.5 + 0.5 * energy) * (1.3 if answering else 1.0):
                vel = (0.78 if s % 8 == 0 else 0.6) + 0.15 * rng.random()
                out.append((s, chord_notes, vel, rng.choice([1, 2, 3]) * rng.uniform(0.55, 0.9)))
        return out

    def arp_bar(self, chord, energy: float) -> list:
        """[(step, midi, vel, dur_steps)] 16th arpeggio over the chord."""
        S = self.steps
        rng = self.rng
        oct_ = self.style["slots"]["arp"].get("octave", 4)
        notes = self._notes(chord, oct_, 4)
        seq = notes + [notes[0] + 12]
        if rng.random() < 0.5:
            seq = seq[::-1]
        step = 1 if energy > 0.6 else 2
        dens = self.style["density"].get("arp", 0.8)
        out = []
        for i, s in enumerate(range(0, S, step)):
            if rng.random() < dens:
                vel = 0.8 if s % 4 == 0 else 0.5 + 0.2 * rng.random()
                out.append((s, seq[i % len(seq)], vel, step * rng.uniform(0.45, 0.85)))
        return out
