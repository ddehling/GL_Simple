"""FluidSynth (SoundFont) voice: a stateful block renderer. Unlike the
analog voices it cannot pre-render a note - FluidSynth is one running
synth - so the rack hands it every event that starts inside the block and
it renders the block in segments split at event offsets (sample-accurate
note-ons). Note-offs are scheduled internally.

Optional dependency: `pip install pyfluidsynth` + libfluidsynth + an .sf2
(Debian: fluid-soundfont-gm -> /usr/share/sounds/sf2/FluidR3_GM.sf2)."""
from __future__ import annotations

import heapq
import os

import numpy as np

from lib.gen import RATE

DEFAULT_SF2 = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/default-GM.sf2",
    "/usr/share/soundfonts/default.sf2",
    "media/soundfonts/default.sf2",
]


def find_soundfont(path=None):
    for p in ([path] if path else []) + DEFAULT_SF2:
        if p and os.path.exists(p):
            return p
    return None


class FluidVoice:
    def __init__(self, soundfont=None, gain=0.6):
        import fluidsynth  # lazy: optional dependency
        sf = find_soundfont(soundfont)
        if sf is None:
            raise FileNotFoundError("no SoundFont found; pass --soundfont")
        self.fs = fluidsynth.Synth(gain=gain, samplerate=float(RATE))
        self.sfid = self.fs.sfload(sf)
        self.path = sf
        self._chan = {}          # slot -> MIDI channel
        self._offs = []          # heap of (abs_sample, chan, key)
        self.clock = 0

    def add_slot(self, slot: str, program: int = 0, bank: int = 0):
        ch = len(self._chan)
        self._chan[slot] = ch
        self.fs.program_select(ch, self.sfid, bank, program)

    def has_slot(self, slot):
        return slot in self._chan

    def _render(self, n):
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        raw = self.fs.get_samples(int(n))
        a = np.asarray(raw)
        if a.dtype != np.float32:
            a = a.astype(np.float32) / 32768.0
        return a.reshape(-1, 2)[:n]

    def render(self, n: int, events) -> np.ndarray:
        """events: NoteEvents with at in [clock, clock+n). Returns (n,2)."""
        c0 = self.clock
        # merge starts and pending offs into one ordered list of (abs, kind, ...)
        todo = [(e.at, 1, e) for e in events if e.slot in self._chan]
        for e in events:
            if e.slot in self._chan:
                heapq.heappush(self._offs, (e.at + e.dur, self._chan[e.slot], int(round(e.pitch))))
        out = np.zeros((n, 2), dtype=np.float32)
        pos = c0
        todo.sort(key=lambda t: t[0])
        ti = 0
        while pos < c0 + n:
            nxt_on = todo[ti][0] if ti < len(todo) else None
            nxt_off = self._offs[0][0] if self._offs else None
            cands = [x for x in (nxt_on, nxt_off) if x is not None and x < c0 + n]
            nxt = min(cands) if cands else c0 + n
            seg = nxt - pos
            if seg > 0:
                out[pos - c0:nxt - c0] = self._render(seg)
                pos = nxt
            while ti < len(todo) and todo[ti][0] <= pos:
                e = todo[ti][2]
                self.fs.noteon(self._chan[e.slot], int(round(e.pitch)), int(max(1, min(127, e.vel * 127))))
                ti += 1
            while self._offs and self._offs[0][0] <= pos:
                _, ch, key = heapq.heappop(self._offs)
                self.fs.noteoff(ch, key)
        self.clock = c0 + n
        return out

    def all_off(self):
        for ch in self._chan.values():
            self.fs.all_notes_off(ch)
        self._offs.clear()
