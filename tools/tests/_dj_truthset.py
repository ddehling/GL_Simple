"""The TRUTH SET for the song reading: songs whose every note is known,
rendered through real instruments (the GM SoundFont via FluidSynth) and
separated with the same demucs model the library uses - so the reader is
measured on stems with real separation artifacts, against notes that are
certain. The old synthetic gate (_dj_instruments_test.py) uses clean
additive tones and the reader scores 1.00 on it while real songs score
0.04-0.34 against a transcriber; this set is what real songs look like
to the reader, with the truth attached.

    python tools/tests/_dj_truthset.py build          # compose + render + separate -> logs/truthset/<song>/
    python tools/tests/_dj_truthset.py eval [names]   # read the demucs stems, score NOTES per stem against the truth
    python tools/tests/_dj_truthset.py eval --true    # ...reading the TRUE stems (no separation): the reader's own ceiling

Each song folder: mix.wav, true_<part>.wav (one per part), stems/<demucs
stem>.wav, truth.json {bpm, first_beat_s, parts: {name: {stem, program,
notes: [[t, dur, midi, vel], ...]}}}. Four songs, four styles, ~100 s each:
an 80s pop track (kit, synth bass, brass stabs, pad, lead), a house track
(four-on-floor, sub bass, piano chords, pluck arpeggio, pad), a rock track
(kit with fills, electric bass, power chords, organ, lead guitar) and a
funk track (swung kit with ghost notes, slap bass, clav, horns, e-piano).
Every part has humanised timing (+-8 ms), velocity accents, section
changes, fills and rests - the things the reader must survive.

Scores (eval): per pitched stem the program's notes against the truth,
onset within 60 ms and the same pitch (F1; also octave-blind); per drum
sound the hits (F1 to the best-matching reading voice); and the waveform
figures of the program's render against the TRUE stems (lib/dj/fidelity),
which no evaluation on a real song can give.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RATE = 44100
ROOT = os.path.join("logs", "truthset")
DRUM = {"kick": 36, "snare": 38, "clap": 39, "hat": 42, "ohat": 46, "ride": 51, "tom_lo": 45, "tom_hi": 48, "crash": 49, "rim": 37, "shaker": 70}

# --------------------------------------------------------------------------
# composing
# --------------------------------------------------------------------------


class Song:
    def __init__(self, name, bpm, bars, seed, swing=0.0):
        self.name, self.bpm, self.bars, self.swing = name, bpm, bars, swing
        self.rng = np.random.default_rng(seed)
        self.beat = 60.0 / bpm
        self.first = 0.5                      # first downbeat (s)
        self.parts = {}                       # name -> {"stem", "program", "bank", "gain", "notes": [[t, dur, midi, vel]]}

    def part(self, name, stem, program, gain=1.0, bank=0):
        self.parts[name] = {"stem": stem, "program": program, "bank": bank, "gain": gain, "notes": []}

    def t(self, bar, step):
        """time of 16th step `step` (float ok) in bar (0-based); swing delays the off-8ths."""
        s = float(step)
        if self.swing and (int(round(s)) % 2 == 1):
            s += self.swing
        return self.first + (bar * 16 + s) * self.beat / 4.0

    def note(self, part, bar, step, midi, dur_steps, vel, human=0.008):
        t = self.t(bar, step) + float(self.rng.normal(0.0, human))
        self.parts[part]["notes"].append([round(max(0.0, t), 4), round(dur_steps * self.beat / 4.0, 4), int(midi), round(float(np.clip(vel, 0.05, 1.0)), 3)])

    def hit(self, part, bar, step, sound, vel, human=0.006):
        self.note(part, bar, step, DRUM[sound], 1, vel, human=human)


def _sections(bars):
    """(bar -> section kind) for a pop-like form."""
    kinds = []
    plan = [("intro", 4), ("verse", 8), ("chorus", 8), ("verse", 8), ("chorus", 8), ("break", 4), ("chorus", 8), ("outro", 4)]
    for k, n in plan:
        kinds += [k] * n
    return (kinds * 2)[:bars]


def pop80(seed=1):
    s = Song("pop80", 118.0, 48, seed)
    s.part("kit", "drums", 0, gain=1.0, bank=128)
    s.part("bass", "bass", 38, gain=0.9)              # synth bass 1
    s.part("brass", "other", 62, gain=0.7)            # synth brass
    s.part("pad", "other", 89, gain=0.45)             # warm pad
    s.part("lead", "other", 81, gain=0.55)            # saw lead
    chords = [(45, [0, 4, 7]), (41, [0, 4, 7]), (43, [0, 4, 7]), (38, [0, 3, 7])]   # A F G Dm (roots as midi)
    sec = _sections(s.bars)
    for b in range(s.bars):
        k = sec[b]
        root, tri = chords[b % 4]
        loud = 1.0 if k == "chorus" else (0.85 if k == "verse" else 0.7)
        # drums
        if k != "break":
            for st in (0, 8):
                s.hit("kit", b, st, "kick", 0.95 * loud)
            if k in ("chorus", "outro"):
                s.hit("kit", b, 10, "kick", 0.7 * loud)
            for st in (4, 12):
                s.hit("kit", b, st, "snare", 0.95 * loud)
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "hat", (0.75 if st % 4 == 0 else 0.5) * loud)
            if k == "chorus":
                for st in (2, 6, 10, 14):
                    s.hit("kit", b, st, "hat", 0.35)
            if b % 8 == 7:                                  # fill
                for st, snd in ((12, "tom_hi"), (13, "tom_hi"), (14, "tom_lo"), (15, "tom_lo")):
                    s.hit("kit", b, st, snd, 0.9)
                s.hit("kit", b + 1 if b + 1 < s.bars else b, 0, "crash", 0.9)
        else:
            s.hit("kit", b, 0, "kick", 0.8); s.hit("kit", b, 8, "kick", 0.6)
        # bass: root on 1, octave on the and, fifth on 3, passing notes
        if k != "intro" or b >= 2:
            s.note("bass", b, 0, root, 3, 0.95)
            s.note("bass", b, 3, root + 12, 1, 0.7)
            s.note("bass", b, 6, root, 2, 0.8)
            s.note("bass", b, 8, root + 7, 3, 0.9)
            s.note("bass", b, 11, root + 12, 1, 0.65)
            s.note("bass", b, 14, root + (10 if b % 4 == 3 else 7), 2, 0.75)
        # brass stabs on the chorus
        if k == "chorus":
            for st in (0, 3, 6, 10):
                for iv in tri:
                    s.note("brass", b, st, root + 12 + iv, 1.5, 0.9 if st in (0, 6) else 0.7)
        # pad: whole-bar chord, verse and break
        if k in ("verse", "break", "intro"):
            for iv in tri:
                s.note("pad", b, 0, root + 12 + iv, 16, 0.6, human=0.0)
        # lead: verse melody with rests
        if k in ("verse", "outro"):
            mel = [0, 4, 7, 9, 7, 4, 2, 0]
            for i, st in enumerate((0, 2, 4, 7, 8, 11, 12, 14)):
                if s.rng.random() < 0.25:
                    continue
                s.note("lead", b, st, root + 24 + mel[(i + b) % 8], 1.5 if st % 4 else 2.5, 0.8 + 0.15 * (st % 4 == 0))
    return s


def house(seed=2):
    s = Song("house", 124.0, 48, seed)
    s.part("kit", "drums", 0, gain=1.0, bank=128)
    s.part("sub", "bass", 39, gain=0.9)               # synth bass 2
    s.part("piano", "other", 4, gain=0.6)             # e-piano
    s.part("pluck", "other", 46, gain=0.5)            # harp (pluck)
    s.part("pad", "other", 91, gain=0.4)              # choir-ish pad
    prog = [(40, [0, 3, 7, 10]), (45, [0, 3, 7, 10]), (43, [0, 4, 7, 11]), (38, [0, 3, 7, 10])]   # Em Am G Dm7
    sec = _sections(s.bars)
    for b in range(s.bars):
        k = sec[b]
        root, ch = prog[(b // 2) % 4]
        if k != "break":
            for st in (0, 4, 8, 12):
                s.hit("kit", b, st, "kick", 1.0)
            for st in (2, 6, 10, 14):
                s.hit("kit", b, st, "ohat", 0.6)
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "hat", 0.45 if st % 4 else 0.6)
            if k in ("chorus", "outro"):
                for st in (4, 12):
                    s.hit("kit", b, st, "clap", 0.9)
            if k == "chorus":
                for st in range(1, 16, 4):
                    s.hit("kit", b, st, "shaker", 0.4)
            if b % 8 == 7:
                for st in (12, 13, 14, 15):
                    s.hit("kit", b, st, "snare", 0.5 + 0.12 * (st - 12))
        else:
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "hat", 0.4)
        # sub bass: offbeat 8ths on the root, with an octave jump
        if k != "intro":
            for st in (2, 6, 10, 14):
                s.note("sub", b, st, root - 12, 1.5, 0.9)
            if k == "chorus":
                s.note("sub", b, 15, root, 1, 0.7)
        # piano chords: syncopated stabs
        if k in ("verse", "chorus", "outro"):
            for st in (0, 3, 6, 11):
                for iv in ch:
                    s.note("piano", b, st, root + 12 + iv, 2, 0.75 if st != 0 else 0.9)
        # pluck arpeggio in the chorus and break: 16ths over the chord
        if k in ("chorus", "break"):
            arp = [root + 24 + iv for iv in ch] + [root + 36]
            for st in range(16):
                if st % 4 == 3 and s.rng.random() < 0.3:
                    continue
                s.note("pluck", b, st, arp[st % len(arp)], 0.8, 0.7 + 0.2 * (st % 4 == 0))
        # pad on the intro/verse/break
        if k in ("intro", "verse", "break"):
            for iv in ch[:3]:
                s.note("pad", b, 0, root + 12 + iv, 16, 0.55, human=0.0)
    return s


def rock(seed=3):
    s = Song("rock", 100.0, 48, seed)
    s.part("kit", "drums", 0, gain=1.0, bank=128)
    s.part("bass", "bass", 33, gain=0.9)              # finger bass
    s.part("guitar", "other", 30, gain=0.6)           # distortion guitar
    s.part("organ", "other", 17, gain=0.4)            # percussive organ
    s.part("leadgtr", "other", 29, gain=0.55)         # overdrive guitar
    prog = [(40, [0, 7, 12]), (36, [0, 7, 12]), (43, [0, 7, 12]), (38, [0, 7, 12])]   # E C G D power chords
    sec = _sections(s.bars)
    for b in range(s.bars):
        k = sec[b]
        root, pw = prog[b % 4]
        if k != "break":
            s.hit("kit", b, 0, "kick", 1.0); s.hit("kit", b, 6, "kick", 0.85); s.hit("kit", b, 8, "kick", 0.95)
            if k == "chorus":
                s.hit("kit", b, 11, "kick", 0.7)
            s.hit("kit", b, 4, "snare", 1.0); s.hit("kit", b, 12, "snare", 1.0)
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "ride" if k == "chorus" else "hat", 0.7 if st % 4 == 0 else 0.5)
            if b % 8 == 7:
                for st, snd in ((8, "snare"), (10, "tom_hi"), (12, "tom_hi"), (13, "tom_lo"), (14, "tom_lo"), (15, "snare")):
                    s.hit("kit", b, st, snd, 0.9)
            if b % 8 == 0 and b:
                s.hit("kit", b, 0, "crash", 0.95)
        # bass: 8ths on the root with a walk-up at the bar end
        if k != "intro":
            for st in range(0, 14, 2):
                s.note("bass", b, st, root - 12, 1.8, 0.85 + 0.1 * (st % 4 == 0))
            s.note("bass", b, 14, root - 12 + (2 if b % 4 != 3 else -2), 2, 0.8)
        # power chords: sustained on the verse, driving 8ths on the chorus
        if k in ("verse", "intro"):
            for iv in pw:
                s.note("guitar", b, 0, root + iv, 8, 0.85, human=0.004)
                s.note("guitar", b, 8, root + iv, 8, 0.75, human=0.004)
        elif k in ("chorus", "outro"):
            for st in range(0, 16, 2):
                for iv in pw:
                    s.note("guitar", b, st, root + iv, 1.6, 0.9 if st % 4 == 0 else 0.75, human=0.004)
        # organ holds on the chorus and break
        if k in ("chorus", "break"):
            for iv in (0, 4, 7):
                s.note("organ", b, 0, root + 12 + iv, 16, 0.6, human=0.0)
        # lead guitar: pentatonic phrases, verse 2 and outro
        if (k == "verse" and b >= 20) or k == "outro":
            pent = [0, 3, 5, 7, 10, 12]
            for st in (0, 3, 4, 6, 8, 11, 12):
                if s.rng.random() < 0.3:
                    continue
                s.note("leadgtr", b, st, root + 24 + pent[int(s.rng.integers(0, len(pent)))], 2.5, 0.85)
    return s


def funk(seed=4):
    s = Song("funk", 104.0, 48, seed, swing=0.18)
    s.part("kit", "drums", 0, gain=1.0, bank=128)
    s.part("slap", "bass", 36, gain=0.9)              # slap bass
    s.part("clav", "other", 7, gain=0.55)
    s.part("horns", "other", 61, gain=0.65)           # brass section
    s.part("epiano", "other", 5, gain=0.45)
    prog = [(40, [0, 4, 7, 10]), (40, [0, 4, 7, 10]), (45, [0, 3, 7, 10]), (43, [0, 4, 7, 10])]   # E7 E7 Am7 G7
    sec = _sections(s.bars)
    for b in range(s.bars):
        k = sec[b]
        root, ch = prog[b % 4]
        if k != "break":
            s.hit("kit", b, 0, "kick", 1.0); s.hit("kit", b, 3, "kick", 0.7); s.hit("kit", b, 8, "kick", 0.9); s.hit("kit", b, 10, "kick", 0.65)
            s.hit("kit", b, 4, "snare", 1.0); s.hit("kit", b, 12, "snare", 1.0)
            for st in (6, 11, 15):                                        # ghost notes
                s.hit("kit", b, st, "snare", 0.25)
            for st in range(16):
                s.hit("kit", b, st, "hat", 0.65 if st % 4 == 0 else (0.45 if st % 2 == 0 else 0.3))
            s.hit("kit", b, 14, "ohat", 0.5)
            if b % 4 == 3:
                for st in (13, 14, 15):
                    s.hit("kit", b, st, "snare", 0.6 + 0.1 * (st - 13))
        else:
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "hat", 0.4)
            s.hit("kit", b, 0, "kick", 0.9)
        # slap bass: syncopated line with octave pops
        if k != "intro":
            line = [(0, 0, 1.5, 0.95), (3, 12, 0.8, 0.8), (4, 0, 1, 0.85), (6, 7, 1, 0.8), (8, 0, 1.5, 0.9), (10, 10, 1, 0.75), (11, 12, 0.8, 0.85), (14, 5, 1, 0.8)]
            for st, iv, d, v in line:
                if s.rng.random() < 0.12:
                    continue
                s.note("slap", b, st, root - 12 + iv, d, v)
        # clav: 16th-note chops
        if k in ("verse", "chorus"):
            for st in range(16):
                if st % 4 == 2 or s.rng.random() < 0.2:
                    continue
                for iv in ch[:3]:
                    s.note("clav", b, st, root + 12 + iv, 0.8, 0.7 + 0.2 * (st % 4 == 0))
        # horns: hits on the chorus, a line in the break
        if k == "chorus":
            for st in (0, 6, 11):
                for iv in ch:
                    s.note("horns", b, st, root + 24 + iv, 1.5 if st else 3, 0.9)
        if k == "break":
            for i, st in enumerate((0, 2, 4, 8, 10, 12)):
                s.note("horns", b, st, root + 24 + ch[i % len(ch)], 2, 0.85)
        # e-piano chords, verse and outro
        if k in ("verse", "outro", "intro"):
            for st in (0, 6, 10):
                for iv in ch:
                    s.note("epiano", b, st, root + 12 + iv, 4, 0.65)
    return s


def dnb(seed=5):
    """Fast breaks: ride pattern with open hats on the off-beats, a snare on 2 and 4 with ghosts,
    a reese-like sub bass, stabs and a pad - a kit where ride, open hat and closed hat all occur."""
    s = Song("dnb", 170.0, 48, seed)
    s.part("kit", "drums", 0, gain=1.0, bank=128)
    s.part("sub", "bass", 39, gain=0.9)
    s.part("stab", "other", 81, gain=0.5)             # saw lead as stabs
    s.part("pad", "other", 90, gain=0.4)              # polysynth pad
    s.part("keys", "other", 2, gain=0.45)             # bright piano
    prog = [(38, [0, 3, 7]), (41, [0, 4, 7]), (43, [0, 3, 7]), (36, [0, 4, 7])]   # Dm F Gm C
    sec = _sections(s.bars)
    for b in range(s.bars):
        k = sec[b]
        root, tri = prog[(b // 2) % 4]
        if k != "break":
            s.hit("kit", b, 0, "kick", 1.0); s.hit("kit", b, 10, "kick", 0.9)
            if k == "chorus":
                s.hit("kit", b, 6, "kick", 0.7)
            s.hit("kit", b, 4, "snare", 1.0); s.hit("kit", b, 12, "snare", 1.0)
            for st in (7, 15):
                s.hit("kit", b, st, "snare", 0.25)
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "ride" if k in ("chorus", "outro") else "hat", 0.6 if st % 4 == 0 else 0.45)
            for st in (2, 10):
                s.hit("kit", b, st, "ohat", 0.55)
            if b % 8 == 7:
                for st in (12, 13, 14, 15):
                    s.hit("kit", b, st, "snare", 0.6 + 0.1 * (st - 12))
        else:
            for st in (0, 8):
                s.hit("kit", b, st, "kick", 0.7)
        if k != "intro":
            # (first written two octaves down: D0 at 18 Hz, under the pitch tracker's floor and under
            # hearing - not a note a song would have)
            s.note("sub", b, 0, root - 12, 6, 0.95)
            s.note("sub", b, 6, root - 12, 2, 0.8)
            s.note("sub", b, 8, root - 12 + 3, 4, 0.85)
            s.note("sub", b, 12, root - 12, 4, 0.9)
        if k in ("chorus", "outro"):
            for st in (0, 3, 6, 10, 12):
                for iv in tri:
                    s.note("stab", b, st, root + 12 + iv, 1, 0.85)
        if k in ("verse", "break", "intro"):
            for iv in tri:
                s.note("pad", b, 0, root + 12 + iv, 16, 0.55, human=0.0)
        if k == "verse":
            for st in (0, 4, 8, 12):
                s.note("keys", b, st + (2 if b % 2 else 0), root + 24 + tri[(st // 4) % 3], 2, 0.7)
    return s


def ballad(seed=6):
    """Slow ballad with a VOCAL-like lead (GM 'voice oohs' on the vocals stem): piano chords, fretless bass,
    strings pad, a light kit with rim and ride."""
    s = Song("ballad", 76.0, 40, seed)
    s.part("kit", "drums", 0, gain=0.9, bank=128)
    s.part("bass", "bass", 35, gain=0.85)             # fretless
    s.part("piano", "other", 0, gain=0.6)
    s.part("strings", "other", 48, gain=0.4)
    s.part("voice", "vocals", 53, gain=0.7)           # voice oohs
    prog = [(45, [0, 3, 7]), (41, [0, 4, 7]), (36, [0, 4, 7]), (43, [0, 4, 7])]   # Am F C G
    sec = _sections(s.bars)
    mel = [12, 15, 17, 19, 17, 15, 12, 10, 12, 15, 19, 22, 19, 17, 15, 12]
    for b in range(s.bars):
        k = sec[b]
        root, tri = prog[b % 4]
        if k not in ("intro", "break"):
            s.hit("kit", b, 0, "kick", 0.85); s.hit("kit", b, 8, "kick", 0.75)
            s.hit("kit", b, 4, "rim" if k == "verse" else "snare", 0.8); s.hit("kit", b, 12, "rim" if k == "verse" else "snare", 0.8)
            for st in range(0, 16, 2):
                s.hit("kit", b, st, "ride" if k == "chorus" else "hat", 0.5 if st % 4 == 0 else 0.35)
            if b % 8 == 7:
                s.hit("kit", b, 14, "tom_lo", 0.7); s.hit("kit", b, 15, "tom_lo", 0.8)
        if k != "intro":
            s.note("bass", b, 0, root - 12, 8, 0.85)
            s.note("bass", b, 8, root - 12 + 7, 4, 0.75)
            s.note("bass", b, 12, root - 12 + (5 if b % 4 == 3 else 7), 4, 0.7)
        for st in (0, 8):
            for iv in tri:
                s.note("piano", b, st, root + 12 + iv, 7, 0.7 if st == 0 else 0.6)
        if k in ("chorus", "break", "outro"):
            for iv in tri:
                s.note("strings", b, 0, root + 24 + iv, 16, 0.5, human=0.0)
        if k in ("verse", "chorus"):
            for i, st in enumerate((0, 3, 6, 8, 11, 14)):
                if s.rng.random() < 0.2:
                    continue
                s.note("voice", b, st, root + 12 + mel[(i + 3 * b) % 16], 2.5 if st in (0, 8) else 2, 0.8)
    return s


SONGS = {"pop80": pop80, "house": house, "rock": rock, "funk": funk, "dnb": dnb, "ballad": ballad}

# --------------------------------------------------------------------------
# rendering + separating
# --------------------------------------------------------------------------


def render_part(part, total_s, soundfont):
    """One part through FluidSynth -> (n,2) float32. A small sequencer:
    note-ons at their sample, note-offs from a heap, audio between."""
    import heapq
    import fluidsynth
    fs = fluidsynth.Synth(gain=0.5, samplerate=float(RATE))
    sfid = fs.sfload(soundfont)
    ch = 9 if part["bank"] == 128 else 0
    fs.program_select(ch, sfid, part["bank"], part["program"])
    n = int(total_s * RATE)
    out = np.zeros((n, 2), dtype=np.float32)
    ons = sorted((int(t * RATE), midi, vel, int(max(1, d * RATE))) for t, d, midi, vel in part["notes"])
    offs = []
    pos, i = 0, 0
    while pos < n:
        nxt_on = ons[i][0] if i < len(ons) else None
        nxt_off = offs[0][0] if offs else None
        cands = [x for x in (nxt_on, nxt_off) if x is not None and x < n]
        nxt = min(cands) if cands else n
        if nxt > pos:
            raw = np.asarray(fs.get_samples(nxt - pos))
            if raw.dtype != np.float32:
                raw = raw.astype(np.float32) / 32768.0
            out[pos:nxt] = raw.reshape(-1, 2)[: nxt - pos]
            pos = nxt
        while i < len(ons) and ons[i][0] <= pos:
            at, midi, vel, dur = ons[i]
            fs.noteon(ch, midi, int(max(1, min(127, round(vel * 127)))))
            heapq.heappush(offs, (at + dur, midi))
            i += 1
        while offs and offs[0][0] <= pos:
            _, midi = heapq.heappop(offs)
            fs.noteoff(ch, midi)
    fs.delete()
    return out * np.float32(part["gain"])


def separate(mix):
    """demucs (the library's default model) on a stereo mix -> {stem: (n,2)}."""
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from lib.dj.stems import DEFAULT_STEM_MODEL
    model = get_model(DEFAULT_STEM_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    audio = torch.from_numpy(np.ascontiguousarray(mix.T)).unsqueeze(0)
    ref = audio.mean(0)
    std = ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, audio / std, device=device, shifts=0, split=True, overlap=0.1, progress=False)[0].cpu().numpy() * float(std)
    return {name: out[i].T.astype(np.float32) for i, name in enumerate(model.sources)}


def build(names=None, out_root=ROOT):
    import soundfile as sf
    from lib.gen.synth.fluid import find_soundfont
    sfp = find_soundfont()
    if sfp is None:
        raise FileNotFoundError("no SoundFont (media/soundfonts/default.sf2)")
    for name in names or SONGS:
        t0 = time.time()
        song = SONGS[name]()
        folder = os.path.join(out_root, name)
        os.makedirs(os.path.join(folder, "stems"), exist_ok=True)
        total_s = song.first + song.bars * 4 * song.beat + 2.0
        mix = np.zeros((int(total_s * RATE), 2), dtype=np.float32)
        true_stems = {}
        for pname, part in song.parts.items():
            a = render_part(part, total_s, sfp)
            sf.write(os.path.join(folder, f"true_{pname}.wav"), a, RATE, subtype="PCM_16")
            true_stems.setdefault(part["stem"], np.zeros_like(mix))
            true_stems[part["stem"]] += a
            mix += a
        peak = float(np.abs(mix).max()) or 1.0
        g = 10 ** (-1.0 / 20) / peak
        mix *= g
        for st, a in true_stems.items():
            sf.write(os.path.join(folder, f"true_stem_{st}.wav"), a * g, RATE, subtype="PCM_16")
        sf.write(os.path.join(folder, "mix.wav"), mix, RATE, subtype="PCM_16")
        stems = separate(mix)
        for st, a in stems.items():
            sf.write(os.path.join(folder, "stems", f"{st}.wav"), a, RATE, subtype="PCM_16")
        truth = {"name": name, "bpm": song.bpm, "first_beat_s": song.first, "bars": song.bars, "swing": song.swing, "gain": g,
                 "parts": {p: {k: v for k, v in d.items()} for p, d in song.parts.items()}}
        with open(os.path.join(folder, "truth.json"), "w", encoding="utf-8") as fh:
            json.dump(truth, fh)
        print(f"{name}: {len(song.parts)} parts, {sum(len(p['notes']) for p in song.parts.values())} notes, {total_s:.0f} s, "
              f"separated in {time.time() - t0:.0f} s -> {folder}")


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def _match(a, b, tol=0.06, octave_blind=False):
    if not a or not b:
        return 0.0, 0.0, 0.0
    ref = sorted(b)
    rt = np.array([t for t, _m in ref]); rm = np.array([m for _t, m in ref])
    used = np.zeros(len(ref), dtype=bool)
    hit = 0
    for t, m in sorted(a):
        lo, hi = np.searchsorted(rt, t - tol), np.searchsorted(rt, t + tol)
        for j in range(lo, hi):
            if used[j]:
                continue
            if (rm[j] == m) if not octave_blind else ((rm[j] - m) % 12 == 0):
                used[j] = True
                hit += 1
                break
    p, r = hit / len(a), hit / len(ref)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load(name, out_root=ROOT, true_stems=False):
    import soundfile as sf
    folder = os.path.join(out_root, name)
    with open(os.path.join(folder, "truth.json"), encoding="utf-8") as fh:
        truth = json.load(fh)
    stems = {}
    for st in ("drums", "bass", "other", "vocals"):
        p = os.path.join(folder, f"true_stem_{st}.wav") if true_stems else os.path.join(folder, "stems", f"{st}.wav")
        if os.path.exists(p):
            stems[st] = sf.read(p, dtype="float32")[0]
    return truth, stems, folder


def evaluate(name, out_root=ROOT, true_stems=False, verbose=True):
    from lib.dj import instruments as INS, songprogram as SP, fidelity as F
    truth, stems, folder = load(name, out_root, true_stems=true_stems)
    beat = 60.0 / truth["bpm"]
    n_beats = int((min(len(a) for a in stems.values()) / RATE - truth["first_beat_s"]) / beat)
    beats = truth["first_beat_s"] + np.arange(n_beats) * beat
    t0 = time.time()
    res = INS.identify(stems, beats, down0=0, period=beat)
    mono = {n: INS._mono(a) for n, a in stems.items()}
    prog = SP.build(res, mono, verbatim_stems=(), chords=False, residual=False)
    dt = time.time() - t0
    out = {"name": name, "seconds": dt}
    # pitched stems: notes
    for stem in ("bass", "other", "vocals"):
        tru = [(n[0], n[2]) for p in truth["parts"].values() if p["stem"] == stem for n in p["notes"]]
        if not tru:
            continue
        if stem == "vocals" and not any(v["stem"] == "vocals" for v in prog["voices"].values()):
            continue                                  # vocals carried as phrases (READ_VOCALS off): no note row
        pn = [(t, int(m)) for vid, v in prog["voices"].items() if v["stem"] == stem for t, m, _d, _v in SP.expand(prog, vid) if m is not None]
        p, r, f = _match(pn, tru)
        _p, _r, f2 = _match(pn, tru, octave_blind=True)
        _p, _r, f3 = _match([(t, 0) for t, _m in pn], [(t, 0) for t, _m in tru])
        _p, _r, f30 = _match(pn, tru, tol=0.03)
        # voice PURITY: a voice's notes from its main part, note-weighted (which sound plays the note)
        parts = {pn_: [(n[0], n[2]) for n in pd["notes"]] for pn_, pd in truth["parts"].items() if pd["stem"] == stem}
        tot = pure = 0
        for vid, v in prog["voices"].items():
            if v["stem"] != stem:
                continue
            vn = [(t, int(m)) for t, m, _d, _v in SP.expand(prog, vid) if m is not None]
            if not vn or not parts:
                continue
            pure += max(_match(vn, pl)[0] * len(vn) for pl in parts.values()); tot += len(vn)
        out[stem] = {"f1": f, "precision": p, "recall": r, "f1_octave_blind": f2, "onset_f1": f3, "f1_30ms": f30,
                     "purity": pure / tot if tot else float("nan"), "n_program": len(pn), "n_truth": len(tru)}
    # drums: per sound, the best reading voice
    hits_true = {}
    for p in truth["parts"].values():
        if p["stem"] == "drums":
            for t, _d, midi, _v in p["notes"]:
                snd = next((k for k, m in DRUM.items() if m == midi), str(midi))
                hits_true.setdefault(snd, []).append((t, 0))
    voices = {vid: [(t, 0) for t, _m, _d, _v in SP.expand(prog, vid)] for vid, v in prog["voices"].items() if v["stem"] == "drums"}
    drums = {}
    for snd, tr in hits_true.items():
        best = max(((vid, _match(ev, tr)) for vid, ev in voices.items()), key=lambda kv: kv[1][2], default=(None, (0, 0, 0)))
        drums[snd] = {"voice": best[0], "f1": best[1][2], "precision": best[1][0], "recall": best[1][1], "n_truth": len(tr)}
    all_true = [h for lst in hits_true.values() for h in lst]
    all_prog = [h for lst in voices.values() for h in lst]
    out["drums"] = {"sounds": drums, "onset_f1": _match(all_prog, all_true)[2], "n_voices": len(voices), "n_sounds": len(hits_true)}
    # waveform: the program's render against the TRUE stems
    import soundfile as sf
    wave = {}
    a, b = 5.0, min(len(mono["drums"]) / RATE - 1.0, 100.0)
    sel = (beats >= a) & (beats < b - beat)
    for stem in ("drums", "bass", "other"):
        p = os.path.join(folder, f"true_stem_{stem}.wav")
        if not os.path.exists(p):
            continue
        ref = sf.read(p, dtype="float32")[0].mean(axis=1)[int(a * RATE):int(b * RATE)]
        ids = [v for v, voice in prog["voices"].items() if voice["stem"] == stem]
        rec = SP.render(prog, {stem: mono[stem]}, ids=ids, t0=a, t1=b, room=False, use_residual=False, limit=None)[:, 0][: len(ref)]
        if len(rec) < len(ref):
            rec = np.concatenate([rec, np.zeros(len(ref) - len(rec), dtype=np.float32)])
        wave[stem] = F.compare(ref, rec, beats[sel] - a, beat, pitched=stem != "drums")
    out["waveform"] = wave
    if verbose:
        print(f"== {name} ({'TRUE stems' if true_stems else 'demucs stems'}; read + program in {dt:.0f} s; {len(prog['voices'])} voices)")
        for stem in ("bass", "other", "vocals"):
            if stem not in out:
                continue
            r = out[stem]
            w = wave.get(stem, {})
            print(f"   {stem:6s} notes F1 {r['f1']:.2f} (P {r['precision']:.2f} R {r['recall']:.2f}; 30 ms {r['f1_30ms']:.2f}) octave-blind {r['f1_octave_blind']:.2f} onsets {r['onset_f1']:.2f} "
                  f"purity {r['purity']:.2f} [{r['n_program']} vs {r['n_truth']} notes]   wave gap {w.get('spectral', float('nan')):.1f} dB level {w.get('level', float('nan')):+.1f} chroma {w.get('chroma_r', float('nan')):.2f}")
        d = out["drums"]
        w = wave.get("drums", {})
        print(f"   drums  onsets F1 {d['onset_f1']:.2f}; {d['n_voices']} voices for {d['n_sounds']} sounds   wave gap {w.get('spectral', float('nan')):.1f} dB level {w.get('level', float('nan')):+.1f}")
        for snd, r in sorted(d["sounds"].items(), key=lambda kv: -kv[1]["n_truth"]):
            print(f"      {snd:7s} {r['n_truth']:4d} hits -> {str(r['voice']):8s} F1 {r['f1']:.2f} (P {r['precision']:.2f} R {r['recall']:.2f})")
    return out


def _part_voice(prog, part_notes, stem):
    """The reading voice that plays this truth part: the one whose notes match it best (weighted precision)."""
    from lib.dj import songprogram as SP
    best = None
    for vid, v in prog["voices"].items():
        if v["stem"] != stem:
            continue
        vn = [(t, int(m)) for t, m, _d, _v in SP.expand(prog, vid) if m is not None] if stem != "drums" else [(t, 0) for t, _m, _d, _v in SP.expand(prog, vid)]
        if not vn:
            continue
        p, r, f = _match(vn, part_notes)
        score = f if stem == "drums" else p * min(1.0, len(vn) / max(len(part_notes), 1))
        if best is None or score > best[0]:
            best = (score, vid)
    return best[1] if best else None


def evaluate_render(name, out_root=ROOT, verbose=True, write=True):
    """THE RENDER GATE: each truth part's TRUE notes played through the reading's
    voice for that part (its extracted sound model), compared with the TRUE part
    stem (lib/dj/fidelity.compare, one global gain fitted per part). The sound
    model is measured apart from the reading's note errors, and the renders are
    written next to the true parts for listening (render_<part>.wav)."""
    global SP
    from lib.dj import instruments as INS, songprogram as SP, fidelity as F
    import soundfile as sf
    truth, stems, folder = load(name, out_root)
    beat = 60.0 / truth["bpm"]
    n_beats = int((min(len(a) for a in stems.values()) / RATE - truth["first_beat_s"]) / beat)
    beats = truth["first_beat_s"] + np.arange(n_beats) * beat
    # the reading is cached per INSTRUMENTS_VERSION: the render gate measures the SOUND MODEL, and a
    # renderer change should be re-measured in seconds, not after six re-reads
    cache = os.path.join(folder, f"reading_v{INS.INSTRUMENTS_VERSION}{os.environ.get('READING_TAG', '')}.json")
    res = None
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            res = json.load(fh)
    if res is None:
        res = INS.identify(stems, beats, down0=0, period=beat)
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
    mono = {n: INS._mono(a) for n, a in stems.items()}
    prog = SP.build(res, mono, verbatim_stems=(), chords=False, residual=False)
    a, b = 5.0, min(len(mono["drums"]) / RATE - 1.0, 100.0)
    sel = (beats >= a) & (beats < b - beat)
    out = {"name": name, "parts": {}}
    step_s = beat / STEPS_PER_BEAT
    for pname, part in truth["parts"].items():
        stem = part["stem"]
        if stem not in mono or stem == "vocals":
            continue
        p = os.path.join(folder, f"true_{pname}.wav")
        if not os.path.exists(p):
            continue
        # the true PART files were written before the mix normalisation (true_stem_* and the mix carry it):
        # bring the part to the stems' level, or the level column reads the song's gain (+8 dB on pop80)
        ref = sf.read(p, dtype="float32")[0].mean(axis=1)[int(a * RATE):int(b * RATE)] * np.float32(truth.get("gain", 1.0))
        # a part with nothing in the window (Slakh parts that enter after 100 s) scored 0.0 dB "perfect" and
        # pulled the medians down: it is not measured
        if not any(a <= n[0] < b for n in part["notes"]) or float(np.sqrt(np.mean(ref.astype(np.float64) ** 2))) < 1e-4:
            continue
        if stem == "drums":
            # each drum sound of the kit part on its own: its best voice plays its true hits
            by_snd = {}
            for t, d, midi, v in part["notes"]:
                snd = next((k for k, m in DRUM.items() if m == midi), str(midi))
                by_snd.setdefault(snd, []).append((t, d, midi, v))
            rec = np.zeros(len(ref), dtype=np.float32)
            played = []
            for snd, notes in by_snd.items():
                vid = _part_voice(prog, [(t, 0) for t, _d, _m, _v in notes], "drums")
                if vid is None:
                    continue
                evs = [[t, None, 1, float(v)] for t, _d, _m, v in notes if a <= t < b]
                y = SP.render_events(prog, vid, evs, t0=a, t1=b)[: len(ref)]
                rec[: len(y)] += y
                played.append(f"{snd}->{vid}")
            label = "kit"
        else:
            vid = _part_voice(prog, [(n[0], n[2]) for n in part["notes"]], stem)
            if vid is None:
                continue
            evs = [[t, int(m), max(1, int(round(d / step_s))), float(v)] for t, d, m, v in part["notes"] if a <= t < b]
            rec = SP.render_events(prog, vid, evs, t0=a, t1=b)[: len(ref)]
            played = [vid]
            label = prog["voices"][vid]["model"]
        if len(rec) < len(ref):
            rec = np.concatenate([rec, np.zeros(len(ref) - len(rec), dtype=np.float32)])

        def _measure(rec):
            # one global gain per part: the level column reports what the render's balance was, the spectral
            # column should judge the SOUND
            r_ref = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)) + 1e-9)
            r_rec = float(np.sqrt(np.mean(rec.astype(np.float64) ** 2)) + 1e-9)
            gain_db = 20 * np.log10(r_ref / r_rec) if r_rec > 1e-8 else 0.0
            rec_g = rec * np.float32(10 ** (gain_db / 20.0)) if r_rec > 1e-8 else rec
            return F.compare(ref, rec_g, beats[sel] - a, beat, pitched=stem != "drums"), gain_db, rec_g
        cmp, gain_db, rec_g = _measure(rec)
        # the OTHER model for a pitched voice (samples <-> additive), so the choice is measured, not trusted
        alt = None
        if stem != "drums" and (prog["_sounds"].get(vid) or {}).get("profile") is not None and label in ("pitched", "additive"):
            other_model = "additive" if label == "pitched" else "pitched"
            if other_model == "additive" or (prog["_sounds"][vid].get("pitches")):
                saved = prog["voices"][vid]["model"]
                prog["voices"][vid]["model"] = other_model
                try:
                    rec2 = SP.render_events(prog, vid, evs, t0=a, t1=b)[: len(ref)]
                    if len(rec2) < len(ref):
                        rec2 = np.concatenate([rec2, np.zeros(len(ref) - len(rec2), dtype=np.float32)])
                    cmp2, _g2, _r2 = _measure(rec2)
                    alt = (other_model, cmp2.get("spectral"))
                finally:
                    prog["voices"][vid]["model"] = saved
        out["parts"][pname] = {"stem": stem, "model": label, "voices": played, "spectral": cmp.get("spectral"), "chroma_r": cmp.get("chroma_r"),
                               "onset_f1": cmp.get("onset_f1"), "raw_level_db": -gain_db, "n_notes": len(part["notes"]),
                               "alt_model": alt[0] if alt else None, "alt_spectral": alt[1] if alt else None}
        if write:
            pk = float(np.abs(rec_g).max()) or 1.0
            sf.write(os.path.join(folder, f"render_{pname}.wav"), np.clip(rec_g / max(pk, 1.0) * 0.98 if pk > 1.0 else rec_g, -1, 1), RATE, subtype="PCM_16")
    if verbose:
        print(f"== {name}: TRUE notes through the extracted voices vs the TRUE parts")
        for pname, r in out["parts"].items():
            alt = f"  [{r['alt_model']} {r['alt_spectral']:.1f} dB]" if r.get("alt_spectral") is not None else ""
            print(f"   {pname:8s} ({r['stem']:5s} {r['model']:8s} {','.join(r['voices'])[:40]:40s}) spectral {r['spectral']:5.1f} dB  "
                  f"chroma {r['chroma_r'] if r['chroma_r'] is not None else float('nan'):.2f}  onset F1 {r['onset_f1'] if r['onset_f1'] is not None else float('nan'):.2f}  "
                  f"render level {r['raw_level_db']:+.1f} dB{alt}")
    return out


STEPS_PER_BEAT = 4


def _apply_flags():
    """DJ_FLAGS="polyreader.CLUSTER=agglo,songprogram.HOLD_ADDITIVE=False": module constants for one run."""
    import importlib
    for kv in [x for x in os.environ.get("DJ_FLAGS", "").split(",") if x.strip()]:
        target, val = kv.split("=", 1)
        mod_name, name = target.strip().rsplit(".", 1)
        mod = importlib.import_module("lib.dj." + mod_name)
        cur = getattr(mod, name)
        if isinstance(cur, bool):
            new = val.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, (int, float)):
            new = type(cur)(float(val))
        elif isinstance(cur, str):
            new = val.strip().strip('"\'')
        else:
            new = eval(val)  # noqa: S307 - a developer's own flag
        setattr(mod, name, new)
        print(f"   [{mod_name}.{name} = {new!r}]")


def main():
    _apply_flags()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0] == "build":
        build(args[1:] or None)
        return 0
    if args and args[0] == "render":
        rows = [evaluate_render(n) for n in (args[1:] or list(SONGS))]
        vals = [r["spectral"] for row in rows for r in row["parts"].values() if r["spectral"] is not None]
        by_stem = {}
        for row in rows:
            for r in row["parts"].values():
                by_stem.setdefault(r["stem"], []).append(r["spectral"])
        print("\n== render gate medians: all parts %.1f dB; " % np.median(vals) + "; ".join(f"{s} {np.median(v):.1f} dB (n {len(v)})" for s, v in by_stem.items()))
        return 0
    names = args[1:] if args and args[0] == "eval" else args
    names = names or list(SONGS)
    true = "--true" in sys.argv
    rows = [evaluate(n, true_stems=true) for n in names]
    print("\n== medians over songs")
    for stem in ("bass", "other", "vocals"):
        rows_s = [r for r in rows if stem in r]
        if not rows_s:
            continue
        print(f"   {stem:6s} notes F1 {np.median([r[stem]['f1'] for r in rows_s]):.2f} (30 ms {np.median([r[stem]['f1_30ms'] for r in rows_s]):.2f})  "
              f"octave-blind {np.median([r[stem]['f1_octave_blind'] for r in rows_s]):.2f}  onsets {np.median([r[stem]['onset_f1'] for r in rows_s]):.2f}  "
              f"purity {np.nanmedian([r[stem]['purity'] for r in rows_s]):.2f}  wave gap {np.median([r['waveform'].get(stem, {}).get('spectral', np.nan) for r in rows_s]):.1f} dB")
    print(f"   drums  onsets F1 {np.median([r['drums']['onset_f1'] for r in rows]):.2f}  per-sound F1 median "
          f"{np.median([s['f1'] for r in rows for s in r['drums']['sounds'].values()]):.2f}  wave gap {np.median([r['waveform'].get('drums', {}).get('spectral', np.nan) for r in rows]):.1f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
