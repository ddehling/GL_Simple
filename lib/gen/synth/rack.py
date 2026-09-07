"""SynthRack: NoteEvents in, 44.1 kHz stereo blocks out. Implements the
AudioEngine track protocol (read/done/fade_out/flags) so it can be
mounted with engine.attach_track() later; today gen_player drives it.

Notes are synthesised at schedule() time on the conductor thread (a
4-bar pad chord is ~100 ms of work; the audio thread only mixes). Per
block: pop due events -> automation events move the mix lanes, notes
join the active list (polyphony cap, drum chokes) -> per-slot gain
(auto gain staging x patch gain x style morph), pan (+ per-note spread),
sends (delay / reverb / chorus); FluidSynth slots render in-block;
per-bus chain (drum compressor, bass saturation, music high-pass), kick
sidechain (depth from the "duck" lane); returns; mix high-pass / low-
pass lanes (the form sweeps them into drops and breaks); master gain;
loudness normalisation toward the style's target; master high shelf;
lookahead limiter; shutdown fade."""
from __future__ import annotations

import heapq
import threading

import numpy as np

from lib.gen import RATE
from lib.gen.events import DRUM_SLOTS
from lib.gen.synth import fx
from lib.gen.synth import plugins
from lib.gen.synth.voices import VOICES

BUS_OF = {s: "drums" for s in DRUM_SLOTS}
BUS_OF.update({"bass": "bass", "lead": "music", "pad": "music", "arp": "music", "keys": "music", "fx": "fx", "vox": "vox",
               "loop_drums": "fx", "loop_bass": "fx", "loop_other": "fx", "loop_vox": "fx",     # loops keep their own mix: no bus chain
               "melody": "music"})
PAN_OF = {"kick": 0.0, "snare": 0.05, "hat": -0.25, "ohat": 0.3, "perc": 0.4, "tom": 0.15,
          "rim": -0.2, "ride": 0.35, "shaker": -0.35,
          "bass": 0.0, "lead": -0.15, "pad": 0.0, "arp": 0.2, "keys": -0.3, "fx": 0.0, "vox": 0.0, "melody": -0.1}
# per-note random pan spread (+-), so repeated hits do not stack on one point
PAN_SPREAD = {"hat": 0.12, "ohat": 0.1, "perc": 0.35, "arp": 0.3, "keys": 0.15, "lead": 0.1,
              "tom": 0.25, "shaker": 0.1, "ride": 0.05}
SQRT2 = float(np.sqrt(2.0))
BUSES = ("drums", "bass", "music", "fx", "vox")     # vox: a song's placed vocal phrases, on their own so they measure as a stem
# Mono-legato drums: a new note in the key slot cuts the tails of notes
# in the listed slots (a 909 kick retriggers, a closed hat chokes the
# open hat). Fade length in seconds.
CHOKE = {"kick": (("kick",), 0.004), "hat": (("ohat",), 0.003), "ohat": (("ohat",), 0.003)}
# Polyphony: per-slot caps (oldest note stolen with a short fade) and a
# total cap, so a dense drop on a slow box cannot pile up buffers.
POLY = {"pad": 12, "keys": 16, "arp": 12, "lead": 8, "bass": 6}
POLY_DEFAULT = 16
POLY_TOTAL = 96
# Mix automation lanes and their resting values. The composer writes
# "auto" events (params lane/to, dur = ramp) from what the form knows;
# the director can move them too (set_lane).
LANES = {"hp": 20.0, "lp": 20000.0, "duck": 0.45, "verb": 1.0, "delay_fb": 0.42, "gain": 1.0}
# gain: a linear multiplier on the mix after the master (before loudness normalisation) - the
# composer writes a song's bar-by-bar dynamics here (SongScript "dyn"), the director can ride it

# Auto gain staging: each slot's voice is rendered once at start-up (a
# reference note, vel 0.8) and trimmed so its RMS lands on the slot's
# target, so a patch's "gain" means the same loudness whichever voice
# class sits behind it. Targets are the measured values of the groove
# style's original voices (2026-09-06), so that balance is the reference.
SLOT_TARGET_DB = {"kick": -10.5, "snare": -14.9, "hat": -17.5, "ohat": -12.9, "perc": -12.1,
                  "tom": -12.0, "rim": -12.5, "ride": -13.5, "shaker": -17.0,
                  "bass": -11.5, "lead": -13.1, "pad": -22.2, "arp": -14.4, "keys": -7.5}
CAL_PITCH = {"kick": 36.0, "snare": 38.0, "hat": 42.0, "ohat": 46.0, "perc": 60.0, "tom": 48.0,
             "rim": 37.0, "ride": 51.0, "shaker": 70.0, "bass": 45.0}
CAL_RNG_SEED = 1234
NO_CAL = ("sample", "fx")


class SynthRack:
    is_narrative = False
    is_ambient = False
    is_soundpool = False

    def __init__(self, style: dict, bpm: float, fluid=None, fluid_slots=(),
                 seed: int = 1, master: float = 0.8):
        style = plugins.overlay(style)      # hosted instruments where the style asks and the plugin exists
        self.style = style
        self.bpm = float(bpm)
        self.bus_fx = self._load_bus_fx(style)
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
        self._pending = []          # heap of (at, seq, NoteEvent, prerendered|None)
        self._lock = threading.Lock()   # step thread schedules, render thread pops
        self._rng_lock = threading.Lock()   # the seeded rng is touched from both threads
        self._seq = 0
        self.muted = set()          # slots silenced at render time (immediate)
        self._style_swap = None     # (at_sample, style, bpm, morph_samples) applied by render()
        self._morph = None          # (t0, t1, {slot: old effective gain})
        self._active = []           # [start, buf, slot, patch, (l, r)]
        self.clock = 0
        self.done = False
        self._fade = None
        self.capture = None         # offline stems: callable(name, block) fed the four buses per render block
        self._gain = 1.0
        self.master = float(master)
        beat = RATE * 60.0 / self.bpm
        self.delay = fx.PingPongDelay(int(beat * 0.75))
        self.reverb = fx.FDNReverb(size=float(style.get("reverb_size", 1.0)), decay=float(style.get("reverb_decay", 0.8)))
        self.chorus = fx.Chorus()
        self.limiter = fx.LookaheadLimiter(ceiling=0.95)
        # per-bus processing: drum glue, bass harmonics, music out of the
        # kick's way, a little air on the master
        self.drum_comp = fx.Compressor(thresh_db=-14.0, ratio=3.0, attack_s=0.004, release_s=0.09, makeup_db=2.5)
        self.bass_sat = fx.Saturator(1.8)
        self.music_hp = fx.Biquad("highpass", 170.0, 0.8)
        self.master_shelf = fx.Biquad("highshelf", 6000.0, 0.7, float(style.get("master_shelf_db", 2.5)))
        self.mix_hp = fx.Biquad("highpass", LANES["hp"], 0.7)
        self.mix_lp = fx.Biquad("lowpass", LANES["lp"], 0.7)
        self.lanes = {k: [float(v), float(v), 0] for k, v in LANES.items()}   # value, target, samples left
        self._lane_applied = {"hp": LANES["hp"], "lp": LANES["lp"]}
        # loudness normalisation toward the style's target (None = off)
        self.loud = fx.Loudness()
        self.target_lufs = style.get("target_lufs")
        self.norm_db = 0.0
        self.trim = {}
        self.calibrate()
        self._kicks = []            # recent kick onsets for the duck
        self.has_kick = "kick" in self.slots
        self.stats = {"notes": 0, "peak": 0.0, "blocks": 0, "stolen": 0}
        # Monitor ring (4 s) for scopes: the render thread writes, a GUI
        # reads - a torn read is a torn picture, never a crash.
        self._mon = np.zeros((RATE * 4, 2), dtype=np.float32)
        self._mon_w = 0

    # -- control-thread API --------------------------------------------------
    def warm_up(self):
        """Compile (or load from cache) every numba kernel the render path
        can hit, BEFORE the audio callback depends on us: a first-call JIT
        inside the render thread is a multi-second stall = ring underrun.
        Renders one short note per voice class and one block of FX with a
        throwaway rng, so the seeded stream is untouched."""
        rng = np.random.default_rng(0)
        n = 256
        for name, vc in VOICES.items():
            if name == "sample":            # nothing to compile, and no file to complain about
                continue
            try:
                vc().render(48.0, 0.7, n, {}, {"glide_from": 47.0} if name == "bass" else {}, rng)
            except Exception:  # noqa: BLE001 - a warm-up must never block a start
                pass
        blk = np.zeros((n, 2), dtype=np.float32)
        fx.PingPongDelay(n).process(blk)
        fx.FDNReverb().process(blk)
        fx.Chorus().process(blk)
        fx.LookaheadLimiter().process(blk)
        fx.Loudness().feed(blk)
        fx.Compressor().process(blk)
        fx.Biquad("highpass", 100.0).process(blk)
        fx.duck_curve(n, 0, [0])
        for kind in ("riser", "revcym", "impact", "sweep"):
            VOICES["fx"]().render(48.0, 0.7, n, {}, {"kind": kind}, rng)

    def schedule(self, events):
        """Queue notes - and render them NOW, on the caller's (conductor)
        thread. Phrases arrive bars ahead, so the whole-note synthesis
        (a 4-bar pad chord is ~100 ms of work) never lands in the audio
        thread; render() only mixes. Notes that were muted at schedule
        time, or whose slot has no analog voice, are rendered on due.
        "auto" events are automation and carry no audio."""
        with self._lock:
            swap = self._style_swap
        events = list(events)
        vst_batches = {}
        for e in events:
            pre = None
            if e.slot != "auto" and e.slot not in self.fluid_slots and e.slot not in self.muted:
                slots = swap[1]["slots"] if (swap is not None and e.at >= swap[0]) else self.slots
                patch = slots.get(e.slot)
                if patch and patch.get("voice") == "vst":
                    # hosted (VST) slots: stateful instruments render the whole
                    # batch of their notes in ONE call plus a release tail
                    vst_batches.setdefault(e.slot, (patch, []))[1].append(e)
                    continue
                vc = VOICES.get(patch.get("voice")) if patch else None
                if vc is not None:
                    pre = self._synth(vc(), e, patch)
            with self._lock:
                self._seq += 1
                heapq.heappush(self._pending, (e.at, self._seq, e, pre))
        for slot, (patch, evs) in vst_batches.items():
            self._schedule_vst(slot, patch, evs)

    def _schedule_vst(self, slot, patch, evs):
        """Render a hosted instrument for this batch as one buffer from
        the first note to the last note-off plus a tail; queue it as a
        single pre-rendered event. Missing plugin -> the fallback patch."""
        inst = plugins.instrument(patch)
        if inst is None:
            fb = patch.get("fallback") or {}
            vc = VOICES.get(fb.get("voice"))
            for e in evs:
                pre = self._synth(vc(), e, fb) if vc is not None else None
                with self._lock:
                    self._seq += 1
                    heapq.heappush(self._pending, (e.at, self._seq, e, pre))
            return
        t0 = min(e.at for e in evs)
        t1 = max(e.at + e.dur for e in evs)
        tail = float(patch.get("tail", 2.5))
        seconds = (t1 - t0) / RATE + tail
        try:
            buf = inst.render(evs, t0, seconds, vel_curve=float(patch.get("vel_curve", 1.0)))
        except Exception as e:  # noqa: BLE001 - a plugin must never take the conductor down
            self.stats["render_errors"] = self.stats.get("render_errors", 0) + 1
            self.stats["last_render_error"] = f"vst {slot}: {type(e).__name__}: {e}"
            return
        start = max(0, t0 - int(inst.latency))
        with self._rng_lock:
            pan = self._pan(slot, {}, True)
        head = evs[0]
        marker = type(head)(start, slot, head.pitch, head.vel, buf.shape[0], {"vst_batch": len(evs)})
        with self._lock:
            self._seq += 1
            heapq.heappush(self._pending, (marker.at, self._seq, marker, (buf, patch, pan)))

    def _load_bus_fx(self, style):
        """style["bus_fx"] = {"master": [{"plugin": "vst:...", "mix": 1.0, "params": {...}}], "music": [...], ...}"""
        out = {}
        for bus, specs in (style.get("bus_fx") or {}).items():
            chain = [f for f in (plugins.effect(sp) for sp in specs) if f is not None]
            if chain:
                out[bus] = chain
        return out

    def _synth(self, voice, e, patch):
        """Render one note: the slot's voice plus any layer patches
        (patch["layers"]: [{"voice": ..., "gain": ..., "hp": Hz, "lp": Hz, ...}]),
        summed into one buffer. Layers are how a bass becomes sub + top
        and a kick becomes body + click."""
        with self._rng_lock:
            buf = voice.render(e.pitch, e.vel, e.dur, patch, e.params, self.rng)
            for layer in patch.get("layers", ()) or ():
                lvc = VOICES.get(layer.get("voice"))
                if lvc is None:
                    continue
                lp = {k: v for k, v in patch.items() if k != "layers"}
                lp.update(layer)
                lb = lvc().render(e.pitch, e.vel, e.dur, lp, e.params, self.rng)
                lb = self._crossover(lb, layer)
                lb = lb * np.float32(layer.get("gain", 0.5))
                buf = self._sum(buf, lb)
            pan = self._pan(e.slot, e.params, buf.ndim == 2)
        return buf, patch, pan

    @staticmethod
    def _crossover(buf, layer):
        from lib.gen.synth import dsp
        hp, lp = float(layer.get("hp", 0.0) or 0.0), float(layer.get("lp", 0.0) or 0.0)
        if not hp and not lp:
            return buf
        chans = [buf] if buf.ndim == 1 else [np.ascontiguousarray(buf[:, c]) for c in range(2)]
        out = []
        for ch in chans:
            n = ch.shape[0]
            if hp:
                ch = dsp.svf_tpt(ch, np.full(n, hp, dtype=np.float32), 0.1, RATE, 2, 1.0)
            if lp:
                ch = dsp.svf_tpt(ch, np.full(n, lp, dtype=np.float32), 0.1, RATE, 0, 1.0)
            out.append(ch)
        return out[0] if buf.ndim == 1 else np.stack(out, axis=1)

    @staticmethod
    def _sum(a, b):
        n = max(a.shape[0], b.shape[0])
        if a.ndim == 1 and b.ndim == 1:
            out = np.zeros(n, dtype=np.float32)
            out[: a.shape[0]] += a
            out[: b.shape[0]] += b
            return out
        out = np.zeros((n, 2), dtype=np.float32)
        for x in (a, b):
            if x.ndim == 1:
                out[: x.shape[0], 0] += x
                out[: x.shape[0], 1] += x
            else:
                out[: x.shape[0]] += x
        return out

    def set_style(self, style: dict, bpm: float, at: int = 0, morph: int = 0):
        """Swap patches (and the tempo-synced delay) once the render clock
        reaches `at` - the composer's phrase boundary - so already-scheduled
        notes finish under the patches they were written for. morph > 0:
        slot gains glide from the old style to the new one over that many
        samples instead of jumping."""
        with self._lock:
            self._style_swap = (int(at), style, float(bpm), int(morph))

    def _apply_style(self, style, bpm, morph=0):
        style = plugins.overlay(style)
        self.bus_fx = self._load_bus_fx(style)
        if morph > 0:
            old = {s: float(p.get("gain", 0.5)) * self.trim.get(s, 1.0) for s, p in self.slots.items()}
            self._morph = (self.clock, self.clock + int(morph), old)
        else:
            self._morph = None
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
        self.target_lufs = style.get("target_lufs", self.target_lufs)
        self.master_shelf = fx.Biquad("highshelf", 6000.0, 0.7, float(style.get("master_shelf_db", 2.5)))
        self.reverb.set(decay=style.get("reverb_decay"))
        self.calibrate()

    def calibrate(self):
        """Measure each slot's voice on a reference note and set the trim
        that puts it on SLOT_TARGET_DB. Deterministic (own rng), cheap
        (one short note per slot), and skipped for slots without a target
        or without an analog voice (fluid, sample, fx)."""
        rng = np.random.default_rng(CAL_RNG_SEED)
        self.trim = {}
        for slot, patch in self.slots.items():
            target = SLOT_TARGET_DB.get(slot)
            vc = VOICES.get(patch.get("voice"))
            pitch = CAL_PITCH.get(slot, 12.0 * (int(patch.get("octave", 3)) + 1) + 9.0)
            if patch.get("voice") == "vst" and target is not None:
                inst = plugins.instrument(patch)
                if inst is None:
                    continue
                from lib.gen.events import NoteEvent
                buf = inst.render([NoteEvent(0, slot, pitch, 0.8, int(0.3 * RATE), {})], 0, 0.45,
                                  vel_curve=float(patch.get("vel_curve", 1.0)))
                head = buf[: int(0.4 * RATE)]
                rms = float(np.sqrt(np.mean(head.astype(np.float64) ** 2))) + 1e-9
                # hosted patches can be very quiet or hot; give the trim more room
                # (rounded: a plugin's first render can differ by a few ulps from the next)
                self.trim[slot] = round(float(np.clip(10 ** ((target - 20.0 * np.log10(rms)) / 20.0), 0.2, 8.0)), 3)
                continue
            if target is None or vc is None or patch.get("voice") in NO_CAL:
                continue
            buf = vc().render(pitch, 0.8, int(0.3 * RATE), patch, {}, rng)
            head = buf[: int(0.4 * RATE)]
            rms = float(np.sqrt(np.mean(head.astype(np.float64) ** 2))) + 1e-9
            db = 20.0 * np.log10(rms)
            self.trim[slot] = float(np.clip(10 ** ((target - db) / 20.0), 0.35, 3.0))

    def slot_calibration(self):
        """{slot: measured dBFS of the reference note at the patch gain
        BEFORE trim} - the survey that fills SLOT_TARGET_DB."""
        rng = np.random.default_rng(CAL_RNG_SEED)
        out = {}
        for slot, patch in self.slots.items():
            vc = VOICES.get(patch.get("voice"))
            if vc is None or patch.get("voice") in NO_CAL:
                continue
            pitch = CAL_PITCH.get(slot, 12.0 * (int(patch.get("octave", 3)) + 1) + 9.0)
            buf = vc().render(pitch, 0.8, int(0.3 * RATE), patch, {}, rng)
            head = buf[: int(0.4 * RATE)]
            out[slot] = round(20.0 * np.log10(float(np.sqrt(np.mean(head.astype(np.float64) ** 2))) + 1e-9), 1)
        return out

    def set_master(self, value: float):
        self.master = float(max(0.0, min(1.0, value)))

    def set_mute(self, slot: str, on: bool):
        (self.muted.add if on else self.muted.discard)(slot)

    def set_lane(self, name: str, to: float, ramp_samples: int = 0):
        """Move a mix lane (hp/lp Hz, duck depth, verb multiplier,
        delay_fb) to `to` over `ramp_samples`. Any thread."""
        lane = self.lanes.get(name)
        if lane is None:
            return
        with self._lock:
            lane[1] = float(to)
            lane[2] = max(0, int(ramp_samples))
            if lane[2] == 0:
                lane[0] = float(to)

    def lane_values(self):
        return {k: round(v[0], 3) for k, v in self.lanes.items()}

    def fade_out(self, duration=1.0):
        self._fade = 1.0 / max(duration, 0.05) / RATE

    def pending_until(self):
        with self._lock:
            return max((e[0] for e in self._pending), default=self.clock)

    # -- render --------------------------------------------------------------
    def _pan(self, slot, params, stereo):
        """Constant-power pan gains for this note. Mono buffers are placed;
        stereo buffers are balanced (centre = unity on both sides)."""
        p = PAN_OF.get(slot, 0.0) + float(params.get("pan", 0.0))
        spread = PAN_SPREAD.get(slot, 0.0)
        if spread:
            p += spread * float(self.rng.uniform(-1.0, 1.0))
        p = max(-1.0, min(1.0, p))
        a = (p + 1.0) * np.pi / 4.0
        l, r = float(np.cos(a)), float(np.sin(a))
        if stereo:
            l *= SQRT2
            r *= SQRT2
        return np.float32(l), np.float32(r)

    def _cut(self, item, at, fade_s):
        start, buf = item[0], item[1]
        if start >= at or start + buf.shape[0] <= at:
            return
        fade = max(int(fade_s * RATE), 8)
        cut = at - start
        k = min(fade, cut)
        new = buf[:cut].copy()
        ramp = np.linspace(1.0, 0.0, k, dtype=np.float32)
        if new.ndim == 2:
            new[cut - k:] *= ramp[:, None]
        else:
            new[cut - k:] *= ramp
        item[1] = new

    def _choke(self, slots, at, fade_s):
        """Truncate active notes of `slots` at sample `at` with a short fade."""
        for item in self._active:
            if item[2] in slots:
                self._cut(item, at, fade_s)

    def _steal(self, slot, at):
        """Polyphony: cut the oldest note of `slot` (or of any slot when the
        total cap is hit) at `at`."""
        cap = POLY.get(slot, POLY_DEFAULT)
        same = [it for it in self._active if it[2] == slot and it[0] < at and it[0] + it[1].shape[0] > at]
        if len(same) >= cap:
            self._cut(min(same, key=lambda it: it[0]), at, 0.006)
            self.stats["stolen"] += 1
        if len(self._active) >= POLY_TOTAL:
            live = [it for it in self._active if it[0] < at and it[0] + it[1].shape[0] > at]
            if live:
                self._cut(min(live, key=lambda it: it[0]), at, 0.006)
                self.stats["stolen"] += 1

    def _slot_gain(self, slot, patch):
        g = float(patch.get("gain", 0.5)) * self.trim.get(slot, 1.0)
        m = self._morph
        if m is not None:
            t0, t1, old = m
            f = min(1.0, max(0.0, (self.clock - t0) / max(1, t1 - t0)))
            new = float(self.slots.get(slot, {}).get("gain", 0.0)) * self.trim.get(slot, 1.0)
            g = old.get(slot, 0.0) * (1.0 - f) + new * f
            if f >= 1.0:
                self._morph = None
        return np.float32(g)

    def _tick_lanes(self, n):
        for lane in self.lanes.values():
            v, t, left = lane
            if left > 0:
                step = min(n, left)
                lane[0] = v + (t - v) * (step / left)
                lane[2] = left - step
                if lane[2] == 0:
                    lane[0] = t
        for name, filt in (("hp", self.mix_hp), ("lp", self.mix_lp)):
            v = self.lanes[name][0]
            if abs(v - self._lane_applied[name]) > 0.01 * max(v, 1.0):
                filt.set("highpass" if name == "hp" else "lowpass", max(10.0, min(20000.0, v)), 0.7)
                self._lane_applied[name] = v

    def _apply_auto(self, e):
        lane = e.params.get("lane")
        if lane in self.lanes and "to" in e.params:
            self.set_lane(lane, float(e.params["to"]), int(e.dur))

    def render(self, n: int) -> np.ndarray:
        c0, c1 = self.clock, self.clock + n
        buses = {b: np.zeros((n, 2), dtype=np.float32) for b in BUSES}
        send_d = np.zeros((n, 2), dtype=np.float32)
        send_r = np.zeros((n, 2), dtype=np.float32)
        send_c = np.zeros((n, 2), dtype=np.float32)
        fluid_events = []
        due = []
        with self._lock:
            if self._style_swap is not None and self._style_swap[0] <= c0:
                _, st, bpm, morph = self._style_swap
                self._style_swap = None
                self._apply_style(st, bpm, morph)
            while self._pending and self._pending[0][0] < c1:
                item = heapq.heappop(self._pending)
                due.append((item[2], item[3]))
        self._tick_lanes(n)
        for e, pre in due:
            if e.slot == "auto":
                self._apply_auto(e)
                continue
            if e.slot in self.muted or e.slot not in self.slots:
                continue
            if e.slot in self.fluid_slots:
                fluid_events.append(e)
                continue
            if pre is None:                      # not pre-rendered: do it now
                v = self.voices.get(e.slot)
                if v is None:
                    continue
                pre = self._synth(v, e, self.slots[e.slot])
            buf, patch, pan = pre
            if "vst_batch" not in e.params:       # a hosted batch is one buffer: no choke / steal games
                choke = CHOKE.get(e.slot)
                if choke is not None:
                    self._choke(choke[0], int(e.at), choke[1])
                self._steal(e.slot, int(e.at))
            # The note carries ITS patch: a style swap must not re-price
            # (or, for a slot the new style lacks, crash on) a tail in flight.
            self._active.append([int(e.at), buf, e.slot, patch, pan])
            self.stats["notes"] += 1
            if e.slot == "kick":
                self._kicks.append(int(e.at))
        keep = []
        for item in self._active:
            start, buf, slot, patch, (l, r) = item
            end = start + buf.shape[0]
            if end <= c0:
                continue
            a = max(start, c0)
            b = min(end, c1)
            if a < b:
                seg = buf[a - start:b - start]
                g = self._slot_gain(slot, patch)
                if seg.ndim == 2:
                    sl = seg[:, 0] * (g * l)
                    sr_ = seg[:, 1] * (g * r)
                else:
                    sl = seg * (g * l)
                    sr_ = seg * (g * r)
                bus = buses[BUS_OF.get(slot, "music")]
                bus[a - c0:b - c0, 0] += sl
                bus[a - c0:b - c0, 1] += sr_
                sd = float(patch.get("send_delay", 0.0))
                sr = float(patch.get("send_reverb", 0.0))
                sc = float(patch.get("send_chorus", 0.0))
                if sd:
                    send_d[a - c0:b - c0, 0] += sl * sd
                    send_d[a - c0:b - c0, 1] += sr_ * sd
                if sr:
                    send_r[a - c0:b - c0, 0] += sl * sr
                    send_r[a - c0:b - c0, 1] += sr_ * sr
                if sc:
                    send_c[a - c0:b - c0, 0] += sl * sc
                    send_c[a - c0:b - c0, 1] += sr_ * sc
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
        # sidechain (depth from the duck lane)
        if self.has_kick:
            self._kicks = [k for k in self._kicks if k > c0 - RATE]
            depth = max(0.0, min(0.9, self.lanes["duck"][0]))
            duck = fx.duck_curve(n, c0, [k for k in self._kicks if k < c1], depth=depth)[:, None]
            buses["bass"] *= duck
            buses["music"] *= (0.5 + 0.5 * duck)
        for bus, chain in self.bus_fx.items():
            if bus in buses:
                for f in chain:
                    buses[bus] = f.process(buses[bus])
        b_drums, b_bass = self.drum_comp.process(buses["drums"]), self.bass_sat.process(buses["bass"])
        b_music, b_fx, b_vox = self.music_hp.process(buses["music"]), buses["fx"], buses["vox"]
        if self.capture is not None:                  # offline: the buses as stems (drums / bass / other / vocals)
            self.capture("drums", b_drums); self.capture("bass", b_bass); self.capture("other", b_music); self.capture("vocals", b_vox)
        mix = b_drums + b_bass + b_music + b_fx + b_vox
        self.delay.fb = float(max(0.0, min(0.85, self.lanes["delay_fb"][0])))
        mix += self.delay.process(send_d) * 0.8
        mix += self.reverb.process(send_r) * (0.55 * float(self.lanes["verb"][0]))
        mix += self.chorus.process(send_c) * 0.7
        if self.lanes["hp"][0] > 25.0:
            mix = self.mix_hp.process(mix)
        if self.lanes["lp"][0] < 19000.0:
            mix = self.mix_lp.process(mix)
        mix *= self.master
        if self.target_lufs is not None:
            self.loud.feed(mix)
            lufs = self.loud.lufs()
            if lufs > -45.0 and self.loud.blocks > 40:
                err = float(self.target_lufs) - lufs - self.norm_db
                self.norm_db = float(np.clip(self.norm_db + np.clip(err * 0.02, -0.15, 0.15) * (n / 1024.0), -8.0, 8.0))
            mix = mix * np.float32(10 ** (self.norm_db / 20.0))
        # the gain lane rides AFTER loudness normalisation: scripted bar-by-bar dynamics are
        # not what the normaliser should undo (it holds the long-term level, this is the phrasing)
        g_lane = float(self.lanes["gain"][0])
        if abs(g_lane - 1.0) > 1e-4:
            mix *= np.float32(max(0.0, g_lane))
        for f in self.bus_fx.get("master", ()):
            mix = f.process(mix)
        out = self.limiter.process(self.master_shelf.process(mix))
        if self._fade is not None:
            ramp = self._gain - self._fade * np.arange(1, n + 1, dtype=np.float32)
            ramp = np.clip(ramp, 0.0, 1.0)
            out *= ramp[:, None]
            self._gain = float(ramp[-1])
            if self._gain <= 0.0:
                self.done = True
        L = self._mon.shape[0]
        w = self._mon_w % L
        end = w + n
        if end <= L:
            self._mon[w:end] = out
        else:
            k = L - w
            self._mon[w:] = out[:k]; self._mon[:end - L] = out[k:]
        self._mon_w += n
        self.clock = c1
        self.stats["blocks"] += 1
        pk = float(np.abs(out).max()) if n else 0.0
        if pk > self.stats["peak"]:
            self.stats["peak"] = pk
        return out

    def recent(self, n):
        """The last n rendered stereo samples (oldest first), for scopes."""
        L = self._mon.shape[0]
        n = int(min(max(1, n), L))
        w = self._mon_w % L
        if n <= w:
            return self._mon[w - n:w].copy()
        return np.concatenate([self._mon[L - (n - w):], self._mon[:w]]).copy()

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
