"""SynthRack: NoteEvents in, 44.1 kHz stereo blocks out. Implements the
AudioEngine track protocol (read/done/fade_out/flags) so it can be
mounted with engine.attach_track() later; today gen_player drives it.

Per block: pop due events -> analog voices render whole notes into
buffers that are mixed as they overlap the block; FluidSynth slots render
in-block; per-slot gain/pan/sends; kick sidechain on bass+music; delay +
reverb returns; soft clip; shutdown fade."""
from __future__ import annotations

import heapq
import threading

import numpy as np

from lib.gen import RATE
from lib.gen.events import DRUM_SLOTS
from lib.gen.synth import fx
from lib.gen.synth.voices import VOICES

BUS_OF = {s: "drums" for s in DRUM_SLOTS}
BUS_OF.update({"bass": "bass", "lead": "music", "pad": "music", "arp": "music", "keys": "music"})
PAN_OF = {"kick": 0.0, "snare": 0.05, "hat": -0.25, "ohat": 0.3, "perc": 0.4,
          "bass": 0.0, "lead": -0.15, "pad": 0.0, "arp": 0.2, "keys": -0.3}


class SynthRack:
    is_narrative = False
    is_ambient = False
    is_soundpool = False

    def __init__(self, style: dict, bpm: float, fluid=None, fluid_slots=(),
                 seed: int = 1, master: float = 0.8):
        self.style = style
        self.bpm = float(bpm)
        self.rng = np.random.default_rng(seed)
        self.slots = style["slots"]
        self.voices = {}
        for name, patch in self.slots.items():
            vc = VOICES.get(patch.get("voice"))
            if vc is not None:
                self.voices[name] = vc()
        self.fluid = fluid
        self.fluid_slots = set(fluid_slots) if fluid else set()
        if self.fluid:
            for s in self.fluid_slots:
                self.fluid.add_slot(s, int(self.slots.get(s, {}).get("fluid_program", 0)))
        self._pending = []          # heap of (at, seq, NoteEvent)
        self._lock = threading.Lock()   # step thread schedules, render thread pops
        self._seq = 0
        self.muted = set()          # slots silenced at render time (immediate)
        self._style_swap = None     # (at_sample, style, bpm) applied by render()
        self._active = []           # [start, buf(mono), slot]
        self.clock = 0
        self.done = False
        self._fade = None
        self._gain = 1.0
        self.master = float(master)
        beat = RATE * 60.0 / self.bpm
        self.delay = fx.PingPongDelay(int(beat * 0.75))
        self.reverb = fx.Reverb()
        self._kicks = []            # recent kick onsets for the duck
        self.has_kick = "kick" in self.slots
        self.stats = {"notes": 0, "peak": 0.0, "blocks": 0}

    # -- control-thread API --------------------------------------------------
    def schedule(self, events):
        with self._lock:
            for e in events:
                self._seq += 1
                heapq.heappush(self._pending, (e.at, self._seq, e))

    def set_style(self, style: dict, bpm: float, at: int = 0):
        """Swap patches (and the tempo-synced delay) once the render clock
        reaches `at` - the composer's phrase boundary - so already-scheduled
        notes finish under the patches they were written for."""
        with self._lock:
            self._style_swap = (int(at), style, float(bpm))

    def _apply_style(self, style, bpm):
        self.style = style
        self.slots = style["slots"]
        self.voices = {}
        for name, patch in self.slots.items():
            vc = VOICES.get(patch.get("voice"))
            if vc is not None:
                self.voices[name] = vc()
        if self.fluid is not None:
            for s in self.fluid_slots:
                if s in self.slots and not self.fluid.has_slot(s):
                    self.fluid.add_slot(s, int(self.slots[s].get("fluid_program", 0)))
        if abs(bpm - self.bpm) > 1e-6:
            self.bpm = float(bpm)
            self.delay = fx.PingPongDelay(int(RATE * 60.0 / self.bpm * 0.75))
        self.has_kick = "kick" in self.slots

    def set_master(self, value: float):
        self.master = float(max(0.0, min(1.0, value)))

    def set_mute(self, slot: str, on: bool):
        (self.muted.add if on else self.muted.discard)(slot)

    def fade_out(self, duration=1.0):
        self._fade = 1.0 / max(duration, 0.05) / RATE

    def pending_until(self):
        with self._lock:
            return max((e[0] for e in self._pending), default=self.clock)

    # -- render --------------------------------------------------------------
    def _pan(self, slot):
        p = PAN_OF.get(slot, 0.0)
        return np.float32(np.cos((p + 1) * np.pi / 4)), np.float32(np.sin((p + 1) * np.pi / 4))

    def render(self, n: int) -> np.ndarray:
        c0, c1 = self.clock, self.clock + n
        buses = {b: np.zeros((n, 2), dtype=np.float32) for b in ("drums", "bass", "music")}
        send_d = np.zeros((n, 2), dtype=np.float32)
        send_r = np.zeros((n, 2), dtype=np.float32)
        fluid_events = []
        due = []
        with self._lock:
            if self._style_swap is not None and self._style_swap[0] <= c0:
                _, st, bpm = self._style_swap
                self._style_swap = None
                self._apply_style(st, bpm)
            while self._pending and self._pending[0][0] < c1:
                due.append(heapq.heappop(self._pending)[2])
        for e in due:
            if e.slot in self.muted or e.slot not in self.slots:
                continue
            if e.slot in self.fluid_slots:
                fluid_events.append(e)
                continue
            v = self.voices.get(e.slot)
            if v is None:
                continue
            patch = self.slots[e.slot]
            buf = v.render(e.pitch, e.vel, e.dur, patch, e.params, self.rng)
            # The note carries ITS patch: a style swap must not re-price
            # (or, for a slot the new style lacks, crash on) a tail in flight.
            self._active.append([int(e.at), buf, e.slot, patch])
            self.stats["notes"] += 1
            if e.slot == "kick":
                self._kicks.append(int(e.at))
        keep = []
        for item in self._active:
            start, buf, slot, patch = item
            end = start + buf.shape[0]
            if end <= c0:
                continue
            a = max(start, c0)
            b = min(end, c1)
            if a < b:
                seg = buf[a - start:b - start]
                g = float(patch.get("gain", 0.5))
                l, r = self._pan(slot)
                bus = buses[BUS_OF.get(slot, "music")]
                bus[a - c0:b - c0, 0] += seg * g * l
                bus[a - c0:b - c0, 1] += seg * g * r
                sd = float(patch.get("send_delay", 0.0))
                sr = float(patch.get("send_reverb", 0.0))
                if sd:
                    send_d[a - c0:b - c0, 0] += seg * g * sd * l
                    send_d[a - c0:b - c0, 1] += seg * g * sd * r
                if sr:
                    send_r[a - c0:b - c0, 0] += seg * g * sr * l
                    send_r[a - c0:b - c0, 1] += seg * g * sr * r
            if end > c1:
                keep.append(item)
        self._active = keep
        if self.fluid is not None:
            self.fluid.clock = c0
            fb = self.fluid.render(n, [e for e in fluid_events if self.fluid.has_slot(e.slot)])
            # fluid slots share one stereo return; use the mean of their gains/sends
            gains = [float(self.slots[s].get("gain", 0.5)) for s in self.fluid_slots] or [0.5]
            srs = [float(self.slots[s].get("send_reverb", 0.3)) for s in self.fluid_slots] or [0.3]
            g = float(np.mean(gains)) * 1.6
            buses["music"] += fb * g
            send_r += fb * g * float(np.mean(srs))
        # sidechain
        if self.has_kick:
            self._kicks = [k for k in self._kicks if k > c0 - RATE]
            duck = fx.duck_curve(n, c0, [k for k in self._kicks if k < c1], depth=0.45)[:, None]
            buses["bass"] *= duck
            buses["music"] *= (0.5 + 0.5 * duck)
        mix = buses["drums"] + buses["bass"] + buses["music"]
        mix += self.delay.process(send_d) * 0.7
        mix += self.reverb.process(send_r) * 0.5
        mix *= self.master
        out = fx.softclip(mix, 1.2)
        if self._fade is not None:
            ramp = self._gain - self._fade * np.arange(1, n + 1, dtype=np.float32)
            ramp = np.clip(ramp, 0.0, 1.0)
            out *= ramp[:, None]
            self._gain = float(ramp[-1])
            if self._gain <= 0.0:
                self.done = True
        self.clock = c1
        self.stats["blocks"] += 1
        pk = float(np.abs(out).max()) if n else 0.0
        if pk > self.stats["peak"]:
            self.stats["peak"] = pk
        return out

    # AudioEngine track protocol
    def read(self, n):
        if self.done:
            return None
        try:
            return self.render(int(n))
        except Exception as e:  # noqa: BLE001 - never take the render thread down
            self.stats["render_errors"] = self.stats.get("render_errors", 0) + 1
            self.stats["last_render_error"] = f"{type(e).__name__}: {e}"
            self.clock += int(n)
            return np.zeros((int(n), 2), dtype=np.float32)
