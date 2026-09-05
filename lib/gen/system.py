"""GenSystem: the generative subsystem's conductor - the sibling of
lib/dj/system.py DJSystem.

Owns a Composer and a SynthRack; mounts the rack on the AudioEngine as
ONE track ("gen_rack"); runs a conductor thread that keeps the composer
LEAD_S seconds ahead of the rack's render clock, drains operator steering
(thread-safe queue, applied at phrase boundaries), publishes status() for
the web page and outstate_keys() for the visuals, logs every phrase to
logs/gen_*.jsonl, and supervises itself (a composer exception reseeds and
continues; repeated failures stop with an error, never silence forever).

Long-run: the set arc CYCLES. Each cycle is a "movement": the arc restarts,
the key drifts to a Camelot neighbour and the dice reseed, so an all-night
run keeps changing without ever running out or repeating a night.

All musical time is the rack's SAMPLE clock; wall time is never used for
scheduling (hand-pumped offline rendering behaves identically)."""
from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from queue import SimpleQueue

from lib.gen import RATE
from lib.gen.composer import Composer
from lib.gen.composer.styles import STYLES
from lib.gen.synth import SynthRack
from lib.gen.theory import parse_key

LEAD_S = 6.0            # composed audio kept ahead of the render head
STEP_S = 0.1            # conductor tick
MAX_ERRORS = 5          # per minute before giving up


class GenSystem:
    def __init__(self, engine=None, style="groove", bpm=None, key="8A", seed=None,
                 soundfont=None, fluid_slots="", set_length_s=3 * 3600.0,
                 energy_bias=0.0, density=1.0, swing=None, master=0.8, muted="",
                 log_dir="logs", threaded=True):
        self.engine = engine
        self.style_name = style if style in STYLES else "groove"
        self.bpm_req = float(bpm) if bpm else None
        self.key_req = key
        self.seed = int(seed) if seed is not None else random.randrange(1, 10 ** 6)
        self.soundfont = soundfont
        self.fluid_slots = fluid_slots
        self.set_length_s = float(set_length_s)
        self.energy_bias = float(energy_bias)
        self.density = float(density)
        self.swing = swing
        self.master = float(master)
        self.muted = set((muted or "").split(",")) - {""}
        self.log_dir = log_dir
        self.threaded = threaded
        self.composer = None
        self.rack = None
        self.fluid = None
        self._q = SimpleQueue()
        self._running = False
        self._thread = None
        self._phrases = deque(maxlen=96)     # recent + upcoming phrases
        self._log_tail = deque(maxlen=40)
        self._log_fh = None
        self._errors = deque(maxlen=MAX_ERRORS)
        self.last_error = ""
        self._t0 = None
        self._end_at = None
        self._movement = 0
        self._drop_wall = 0.0
        self._render_t = 0.0                 # cumulative render seconds (cpu)
        self._render_samples = 0
        self._hold = False

    # -- lifecycle --------------------------------------------------------
    def start(self) -> bool:
        if self._running:
            return True
        try:
            self._build()
        except Exception as e:  # noqa: BLE001
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[GEN] start failed: {self.last_error}")
            return False
        self._open_log()
        self._t0 = time.time()
        self._compose_ahead(int(LEAD_S * RATE))
        if self.engine is not None:
            self.engine.attach_track("gen_rack", self.rack)
        self._running = True
        if self.threaded:
            self._thread = threading.Thread(target=self._run, daemon=True, name="gen-conductor")
            self._thread.start()
        print(f"[GEN] started: style={self.style_name} bpm={self.composer.bpm:.1f} "
              f"key={self.composer.key.name} seed={self.seed}"
              + (f" fluid={sorted(self.rack.fluid_slots)}" if self.rack.fluid_slots else ""))
        return True

    def _build(self):
        style = STYLES[self.style_name]
        self.composer = Composer(self.style_name, bpm=self.bpm_req, key=self.key_req,
                                 seed=self.seed, arc_fn=self._arc_fn())
        if self.swing is not None:
            self.composer.swing = float(self.swing)
        self.composer.energy_bias = self.energy_bias
        self.composer.density = self.density
        self.composer.muted = set(self.muted)
        slots = tuple(s for s in (self.fluid_slots or "").split(",") if s in style["slots"])
        fluid = None
        if slots:
            try:
                from lib.gen.synth.fluid import FluidVoice
                fluid = FluidVoice(self.soundfont)
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] SoundFont slots unavailable ({e}); analog fallback")
                slots = ()
        self.fluid = fluid
        self.rack = SynthRack(style, self.composer.bpm, fluid=fluid, fluid_slots=slots,
                              seed=self.seed, master=self.master)
        for s in self.muted:
            self.rack.set_mute(s, True)

    def stop(self, fade_s=1.5):
        if not self._running:
            return
        self._running = False
        if self.rack is not None:
            self.rack.fade_out(fade_s)
        self._log({"event": "stop", "t": self._elapsed()})
        if self._log_fh:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None

    @property
    def active(self):
        return self._running

    def _run(self):
        while self._running:
            self.step()
            time.sleep(STEP_S)

    # -- conducting -------------------------------------------------------
    def step(self):
        """One conductor tick: apply steering, keep the composer ahead,
        detect movement/end. Safe to call by hand (offline tests)."""
        if not self._running:
            return
        try:
            while not self._q.empty():
                fn = self._q.get_nowait()
                fn()
            self._compose_ahead(self.rack.clock + int(LEAD_S * RATE))
            if self._end_at is not None and self.rack.clock >= self._end_at:
                print("[GEN] outro finished - stopping")
                self.stop(fade_s=3.0)
        except Exception as e:  # noqa: BLE001
            now = time.time()
            self._errors.append(now)
            self.last_error = f"{type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()
            self._log({"event": "error", "error": self.last_error, "t": self._elapsed()})
            if len(self._errors) >= MAX_ERRORS and now - self._errors[0] < 60.0:
                print(f"[GEN] {MAX_ERRORS} conductor errors in a minute - stopping")
                self.stop(fade_s=1.0)
                return
            # supervision: fresh dice, keep playing
            try:
                self.composer.reseed(random.randrange(1, 10 ** 6))
            except Exception:
                pass

    def _compose_ahead(self, until_sample):
        c = self.composer
        while c.clock < until_sample:
            p = c.next_phrase()
            self.rack.schedule(p.events)
            self._phrases.append(p)
            self._on_phrase(p)

    def _on_phrase(self, p):
        rec = {"event": "phrase", "bar": p.bar0, "section": p.section,
               "energy": round(p.energy, 3), "chords": [c[1] for c in p.chords],
               "key": p.meta.get("camelot"), "bpm": round(p.bpm, 2),
               "lead": p.meta.get("lead_op"), "layers": p.meta.get("layers"),
               "at_s": round(p.start / RATE, 3), "movement": self._movement,
               "t": self._elapsed()}
        self._log_tail.append(rec)
        self._log(rec)
        f = self.composer.form
        # End-of-set detection: ending requested and the outro has run its
        # course (the form re-drew 'outro' after an outro) -> stop after it.
        if f.ending and self._end_at is None:
            outros = [h for h in f.history if h[1] == "outro"]
            if len(outros) >= 2:
                self._end_at = p.end
        # Movement boundary: the arc cycled -> drift key, reseed, log.
        mv = self._movement_index(p.bar0 + p.nbars)
        if mv != self._movement and not f.ending:
            self._movement = mv
            nk = random.Random(self.seed + mv).choice(self.composer.key.neighbours())
            self.composer.set_key(nk)
            self.composer.reseed(self.seed + 7919 * mv)
            self._log({"event": "movement", "n": mv, "key": nk.name, "t": self._elapsed()})
            print(f"[GEN] movement {mv}: key -> {nk.name} ({nk.camelot})")

    def _bar_seconds(self):
        return 4 * 60.0 / (self.composer.bpm if self.composer else 120.0)

    def _movement_index(self, bar):
        return int(bar * self._bar_seconds() // max(60.0, self.set_length_s))

    def _arc_fn(self):
        def f(bar):
            secs = bar * self._bar_seconds()
            prog = (secs % max(60.0, self.set_length_s)) / max(60.0, self.set_length_s)
            if prog < 0.66:
                return 0.35 + 0.65 * (prog / 0.66)
            return max(0.2, 1.0 - (prog - 0.66) / 0.34 * 0.8)
        return f

    # -- steering (any thread; applied on the conductor) ------------------
    def _post(self, fn):
        self._q.put(fn)

    def set_style(self, name):
        if name not in STYLES:
            return

        def do():
            c = self.composer
            at = c.clock                       # next phrase boundary
            nc = Composer(name, bpm=None, key=c.key, seed=c.seed + 1, arc_fn=self._arc_fn())
            nc.clock, nc.bar = c.clock, c.bar
            nc.energy_bias, nc.density, nc.muted = c.energy_bias, c.density, set(c.muted)
            nc.form.hold, nc.form.ending = c.form.hold, c.form.ending
            if self.swing is not None:
                nc.swing = float(self.swing)
            if self.bpm_req:
                nc.bpm = self.bpm_req
            self.composer = nc
            self.style_name = name
            self.rack.set_style(STYLES[name], nc.bpm, at=at)
            self._log({"event": "style", "style": name, "t": self._elapsed()})
        self._post(do)

    def set_bpm(self, bpm):
        self.bpm_req = float(bpm)

        def do():
            self.composer.set_bpm(self.bpm_req)
            self.rack.set_style(self.composer.style, self.bpm_req, at=self.composer.clock)
        self._post(do)

    def set_key(self, key):
        self.key_req = key
        self._post(lambda: self.composer.set_key(key))

    def set_energy_bias(self, v):
        self.energy_bias = float(v)
        self._post(lambda: setattr(self.composer, "energy_bias", float(v)))

    def set_density(self, v):
        self.density = float(v)
        self._post(lambda: setattr(self.composer, "density", float(v)))

    def set_swing(self, v):
        self.swing = float(v)
        self._post(lambda: setattr(self.composer, "swing", float(v)))

    def set_master(self, v):
        self.master = float(v)
        if self.rack is not None:
            self.rack.set_master(v)

    def set_mute(self, slot, on):
        (self.muted.add if on else self.muted.discard)(slot)
        if self.rack is not None:
            self.rack.set_mute(slot, on)          # immediate
        self._post(lambda: (self.composer.muted.add if on else self.composer.muted.discard)(slot))

    def set_hold(self, on):
        self._hold = bool(on)
        self._post(lambda: setattr(self.composer.form, "hold", bool(on)))

    def reseed(self, seed=None):
        s = int(seed) if seed is not None else random.randrange(1, 10 ** 6)
        self.seed = s
        self._post(lambda: (self.composer.reseed(s), self._log({"event": "reseed", "seed": s, "t": self._elapsed()})))

    def set_set_length(self, seconds):
        self.set_length_s = float(seconds)
        self._post(lambda: self.composer.set_arc(self._arc_fn()))

    def request_end(self):
        self._post(lambda: (self.composer.request_end(), self._log({"event": "end_requested", "t": self._elapsed()})))

    # -- status -----------------------------------------------------------
    def _heard_clock(self):
        """Render head minus the engine's render-ahead lead = what the room
        hears now (same correction DJSystem makes for its visuals)."""
        if self.rack is None:
            return 0
        lead = 0
        if self.engine is not None:
            try:
                lead = int(self.engine.render_lead_frames())
            except Exception:
                lead = 0
        return max(0, self.rack.clock - lead)

    def _phrase_at(self, sample):
        cur = None
        for p in self._phrases:
            if p.start <= sample < p.end:
                cur = p
                break
        return cur

    def _elapsed(self):
        return round(time.time() - self._t0, 2) if self._t0 else 0.0

    def status(self):
        c = self.composer
        if c is None or self.rack is None:
            return {"available": True, "active": False, "state": "idle", "error": self.last_error}
        heard = self._heard_clock()
        p = self._phrase_at(heard)
        spb = c.samples_per_bar
        beat = c.samples_per_beat
        bar_in = ((heard - p.start) // spb if p else 0)
        pos_in_bar = ((heard - p.start) % spb if p else 0)
        f = c.form
        drop_eta = None
        for q in self._phrases:
            for d in q.drops():
                if d > heard:
                    drop_eta = (d - heard) / RATE
                    break
            if drop_eta is not None:
                break
        nxt = None
        try:
            # the form's next section is random; show the possibilities
            nxt = [n for n, _ in c.style["form"][f.section]]
        except Exception:
            pass
        sec_total = f.bars_in + f.bars_left
        stats = self.rack.stats
        return {
            "available": True, "active": self._running,
            "state": "ending" if f.ending else ("hold" if f.hold else "playing"),
            "style": self.style_name,
            "styles": [{"id": k, "label": v["label"], "bpm": list(v["bpm"])} for k, v in STYLES.items()],
            "bpm": round(c.bpm, 2), "key": c.key.name, "camelot": c.key.camelot,
            "mode": c.key.mode,
            "section": p.section if p else f.section,
            "section_next": nxt,
            "section_bars_left": f.bars_left, "section_bars": sec_total,
            "bar": (p.bar0 + bar_in) if p else c.bar,
            "beat": int(pos_in_bar // beat) + 1,
            "bar_phase": float(pos_in_bar / spb),
            "beat_phase": float((pos_in_bar % beat) / beat),
            "phrase_phase": float((heard - p.start) / (p.end - p.start)) if p else 0.0,
            "energy": round(p.energy, 3) if p else round(f.energy(), 3),
            "energy_bias": self.energy_bias, "density": self.density,
            "swing": round(c.swing, 3), "master": self.master,
            "chords": [x[1] for x in p.chords] if p else [],
            "chord_now": (p.chords[min(int(bar_in), len(p.chords) - 1)][1] if p else ""),
            "layers": p.meta.get("layers", []) if p else [],
            "lead_op": p.meta.get("lead_op") if p else None,
            "muted": sorted(self.muted),
            "slots": list(c.style["slots"].keys()),
            "fluid_slots": sorted(self.rack.fluid_slots),
            "drop_eta": round(drop_eta, 2) if drop_eta is not None else None,
            "movement": self._movement,
            "arc_progress": round(((heard / RATE) % max(60.0, self.set_length_s)) / max(60.0, self.set_length_s), 4),
            "set_length_s": self.set_length_s,
            "seed": self.seed, "uptime_s": self._elapsed(),
            "heard_s": round(heard / RATE, 2),
            "notes": stats["notes"], "peak": round(stats["peak"], 3),
            "motifs": len(c.melody.memory),
            "lead_s": round((self.rack.pending_until() - self.rack.clock) / RATE, 1),
            "log": list(self._log_tail)[-14:],
            "error": self.last_error,
        }

    def outstate_keys(self):
        """Published into outstate each tick for the visuals."""
        if not self._running or self.composer is None:
            return {"gen_active": False}
        s = self.status()
        drop_t = None
        if s["drop_eta"] is not None and s["drop_eta"] < 0.05:
            self._drop_wall = time.time()
        return {"gen_active": True, "gen_energy": s["energy"], "gen_bpm": s["bpm"],
                "gen_beat_phase": s["beat_phase"], "gen_bar_phase": s["bar_phase"],
                "gen_phrase_phase": s["phrase_phase"], "gen_section": s["section"],
                "gen_next_drop_eta": s["drop_eta"], "gen_drop_t": self._drop_wall or drop_t,
                "gen_key": s["camelot"]}

    # -- log --------------------------------------------------------------
    def _open_log(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            name = time.strftime("gen_%Y%m%d_%H%M%S.jsonl")
            self._log_fh = open(os.path.join(self.log_dir, name), "a", encoding="utf-8")
            self._log({"event": "start", "style": self.style_name, "seed": self.seed,
                       "bpm": self.composer.bpm, "key": self.composer.key.name,
                       "set_length_s": self.set_length_s, "fluid": sorted(self.rack.fluid_slots)})
        except Exception as e:  # noqa: BLE001
            print(f"[GEN] log disabled: {e}")
            self._log_fh = None

    def _log(self, rec):
        if self._log_fh is None:
            return
        try:
            self._log_fh.write(json.dumps(rec, default=str) + "\n")
            self._log_fh.flush()
        except Exception:
            pass
