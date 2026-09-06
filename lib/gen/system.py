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
from lib.gen.feedback import PreferenceMemory
from lib.gen.composer.hooks import HookAuthor
from lib.gen.scenes import SceneStore, scene_intent, snapshot as scene_snapshot
from lib.gen.synth import SynthRack
from lib.gen.theory import parse_key

LEAD_S = 6.0            # composed audio kept ahead of the render head


def _gesture_menu():
    from lib.gen.director import GESTURES
    return [{"id": k, "label": v["label"]} for k, v in GESTURES.items()]


def _director_available():
    try:
        from lib.gen.director import find_claude_exe
        if find_claude_exe():
            return True
        import anthropic  # noqa: F401
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    except Exception:
        return False


def _strudel_available():
    try:
        from lib.gen.composer.strudel import available
        return bool(available()[0])
    except Exception:
        return False
STEP_S = 0.1            # conductor tick
MAX_ERRORS = 5          # per minute before giving up


class GenSystem:
    def __init__(self, engine=None, style="groove", bpm=None, key="8A", seed=None,
                 soundfont=None, fluid_slots="", set_length_s=3 * 3600.0,
                 energy_bias=0.0, density=1.0, swing=None, master=0.8, muted="", hooks=None,
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
        self._strudel = None                 # StrudelBridge, started on first pattern
        self._pattern_code = None
        self._slot_patterns = {}             # slot -> code (status)
        self.brightness = 1.0
        self._ramps = {}                     # param -> {"from","to","start_bar","bars"}
        self.prefs = PreferenceMemory(os.path.join(log_dir, "gen_prefs.json"))
        # authored hooks: an LLM writes the movement's theme in the background
        # (default: on unless GEN_HOOKS=0 - the gates run with it off so they never call the CLI)
        if hooks is None:
            hooks = os.environ.get("GEN_HOOKS", "1") != "0"
        self.hooks = HookAuthor(os.path.join(log_dir, "gen_hooks.json"), enabled=bool(hooks))
        self.scenes = SceneStore(os.path.join(log_dir, "gen_scenes.json"))
        self.director = None                 # LLMDirector, created on first ask
        self._director_busy = False
        self._director_last = {}             # {"text","say","done","warn","error","t"}
        self._director_log = deque(maxlen=20)

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
        self.rack.warm_up()          # JIT before the audio callback needs blocks
        self.composer.form.taste = self.prefs.section_bias(self.style_name)
        self._wire_hooks(self.composer)
        for s in self.muted:
            self.rack.set_mute(s, True)

    def stop(self, fade_s=1.5):
        if not self._running:
            return
        self._running = False
        if self._strudel is not None:
            self._strudel.stop()
            self._strudel = None
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

    def _tick_ramps(self, next_bar):
        """Advance parameter ramps to the value they should have at
        `next_bar` (the phrase about to be composed)."""
        for k, r in list(self._ramps.items()):
            prog = min(1.0, (next_bar - r["start_bar"]) / max(1, r["bars"]))
            v = r["from"] + (r["to"] - r["from"]) * prog
            self._apply_param(k, v)
            if prog >= 1.0:
                del self._ramps[k]

    def _apply_param(self, k, v):
        c = self.composer
        if k == "energy_bias":
            self.energy_bias = c.energy_bias = float(v)
        elif k == "density":
            self.density = c.density = float(v)
        elif k == "swing":
            self.swing = c.swing = float(v)
        elif k == "brightness":
            self.brightness = c.brightness = float(v)
        elif k == "bpm":
            self.bpm_req = float(v)
            c.set_bpm(float(v))
            self.rack.set_style(c.style, float(v), at=c.clock)
        elif k == "master":
            self.set_master(v)

    def _on_phrase(self, p):
        self._tick_ramps(p.bar0 + p.nbars)
        rec = {"event": "phrase", "bar": p.bar0, "section": p.section,
               "energy": round(p.energy, 3), "chords": [c[1] for c in p.chords],
               "key": p.meta.get("camelot"), "bpm": round(p.bpm, 2),
               "lead": p.meta.get("lead_op"), "layers": p.meta.get("layers"),
               "at_s": round(p.start / RATE, 3), "movement": self._movement,
               "t": self._elapsed()}
        self._log_tail.append(rec)
        self._log(rec)
        f = self.composer.form
        # a build is starting: ask for a hook written over THIS build's chords
        if p.section == "build" and self.hooks.enabled and p.meta.get("lead_op") in ("theme_make", None) and f.bars_in <= p.nbars:
            self.hooks.request(self.style_name, self.composer.style.get("label", self.style_name), self.composer.bpm,
                               self.composer.key.name, n=2, chords=[c[1] for c in p.chords])
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
            if self.hooks.enabled:
                self.hooks.request(self.style_name, self.composer.style.get("label", self.style_name), self.composer.bpm, nk.name, n=3)
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
            nc.pattern_source = c.pattern_source
            if nc.pattern_source is not None:
                nc.pattern_source.slots = set(STYLES[name]["slots"].keys())
            nc.humanize = getattr(c, "humanize", 1.0)
            nc.automation = getattr(c, "automation", True)
            nc.form.taste = self.prefs.section_bias(name)
            self._wire_hooks(nc)
            # style MORPH: slot gains glide over 8 bars instead of jumping
            self.rack.set_style(STYLES[name], nc.bpm, at=at, morph=int(8 * self.rack_bar_samples()))
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

    def set_brightness(self, v):
        self.brightness = float(v)
        self._post(lambda: setattr(self.composer, "brightness", float(v)))

    def add_ramp(self, param, to, bars):
        def do():
            cur = {"energy_bias": self.composer.energy_bias, "density": self.composer.density,
                   "swing": self.composer.swing, "brightness": self.composer.brightness,
                   "bpm": self.composer.bpm}.get(param)
            if cur is None:
                return
            self._ramps[param] = {"from": float(cur), "to": float(to), "start_bar": self.composer.bar, "bars": int(bars)}
            self._log({"event": "ramp", "param": param, "to": to, "bars": bars, "t": self._elapsed()})
        self._post(do)

    def request_section(self, name):
        self._post(lambda: (self.composer.form.request(name),
                            self._log({"event": "section_request", "section": name, "t": self._elapsed()})))

    def set_slot_pattern(self, slot, code):
        """Strudel notes for ONE slot; the rules keep every other slot."""
        def do():
            from lib.gen.composer.strudel import StrudelSource, open_engine
            try:
                if self._strudel is None:
                    self._strudel = open_engine()
                src = StrudelSource(self._strudel, {slot})
                src.load(code)
                self.composer.slot_patterns[slot] = src
                self._slot_patterns[slot] = code
                self._log({"event": "slot_pattern", "slot": slot, "code": code[:300], "t": self._elapsed()})
            except Exception as e:  # noqa: BLE001
                self.last_error = f"pattern[{slot}]: {type(e).__name__}: {e}"
        self._post(do)

    def clear_slot_pattern(self, slot=None):
        def do():
            for s_ in ([slot] if slot else list(self.composer.slot_patterns)):
                self.composer.slot_patterns.pop(s_, None)
                self._slot_patterns.pop(s_, None)
        self._post(do)

    # -- taste ------------------------------------------------------------
    def _wire_hooks(self, c):
        """Give the melody this style's authored hooks and ask for fresh
        ones (in the background) for the current key."""
        c.melody.hook_provider = self.hooks.provider(self.style_name)
        if self.hooks.enabled and self.hooks.count(self.style_name) < 8:
            self.hooks.request(self.style_name, c.style.get("label", self.style_name), c.bpm, c.key.name,
                               n=4, hint=f"Section energy runs intro -> groove -> build -> drop; the hook is the drop's theme.")

    def rack_bar_samples(self):
        return int(RATE * 4 * 60.0 / (self.composer.bpm if self.composer else 120.0))

    def load_script(self, path_or_dict):
        """Follow a SongScript from the next phrase (switching style first
        when the script asks for another one)."""
        from lib.gen import script as _script
        sc = _script.load(path_or_dict) if isinstance(path_or_dict, str) else _script.normalize(path_or_dict)

        def do():
            if sc.get("style") in STYLES and sc["style"] != self.style_name:
                self.set_style(sc["style"])          # posts its own do(); our load runs after it
                self._post(lambda: self.composer.load_script(sc))
            else:
                self.composer.load_script(sc)
            if sc.get("bpm"):
                self.rack.set_style(self.composer.style, float(sc["bpm"]), at=self.composer.clock)
            self._log({"event": "script", "title": sc.get("title"), "sections": len(sc["sections"]), "t": self._elapsed()})
        self._post(do)

    def set_humanize(self, v):
        def do():
            self.composer.humanize = float(v)
        self._post(do)

    def set_automation(self, on):
        def do():
            self.composer.automation = bool(on)
        self._post(do)

    def set_lane(self, lane, to, ramp_s=0.0):
        if self.rack is not None:
            self.rack.set_lane(lane, float(to), int(float(ramp_s) * RATE))

    # -- timeline ---------------------------------------------------------
    def timeline(self, past_s=240.0):
        """What has played, what is composed ahead, and what the form
        knows beyond that - for the console's Timeline tab.
          now_s      playback position (rack clock) in seconds
          phrases    [{bar0, nbars, start_s, end_s, section, energy, chords,
                       key, lead, layers, drops:[s], played: bool}] from
                       past_s ago up to the last composed phrase
          horizon    {section, bars_left, end_s, next: [(section, weight)],
                       requested, hold, ending, drop_s, movement,
                       set_length_s, arc: [(s, energy)] for the next 10 min}
        """
        if self.rack is None or self.composer is None:
            return {"now_s": 0.0, "phrases": [], "horizon": {}}
        now = self.rack.clock
        bar_s = self._bar_seconds()
        out = []
        for p in list(self._phrases):
            if p.end < now - past_s * RATE:
                continue
            out.append({"bar0": p.bar0, "nbars": p.nbars, "start_s": round(p.start / RATE, 3), "end_s": round(p.end / RATE, 3),
                        "section": p.section, "energy": round(p.energy, 3), "chords": [c[1] for c in p.chords],
                        "key": p.meta.get("camelot"), "lead": p.meta.get("lead_op"), "layers": p.meta.get("layers", []),
                        "drops": [round(d / RATE, 3) for d in p.drops()], "played": p.end <= now,
                        "auto": [{"lane": e.params.get("lane"), "to": e.params.get("to")} for e in p.events if e.slot == "auto"][:6]})
        f = self.composer.form
        c = self.composer
        comp_end_s = c.clock / RATE                       # end of the last composed phrase
        grammar = f.style["form"].get(f.section, [])
        nxt = []
        tot = sum(w for _, w in grammar) or 1.0
        arc = float(f.arc_fn(f._bar))
        for name, w in grammar:
            e = f.style["sections"][name]["energy"]
            bias = 1.0 + 1.5 * (arc - 0.5) * (e - 0.5) * 2.0
            nxt.append((name, round(max(0.05, w * bias * float(f.taste.get(name, 1.0))) / tot, 3)))
        nxt.sort(key=lambda x: -x[1])
        drop_bar = f.upcoming_drop_bar()
        horizon = {"section": f.section, "bars_in": f.bars_in, "bars_left": f.bars_left,
                   "section_end_s": round(comp_end_s + f.bars_left * bar_s, 3),
                   "composed_to_s": round(comp_end_s, 3), "next": nxt,
                   "requested": f.requested, "hold": f.hold, "ending": f.ending,
                   "drop_s": round(comp_end_s + (drop_bar - c.bar) * bar_s, 3) if drop_bar is not None else None,
                   "movement": self._movement, "set_length_s": self.set_length_s,
                   "arc": [(round(comp_end_s + k * 30.0, 1), round(float(f.arc_fn(c.bar + int(k * 30.0 / bar_s))), 3)) for k in range(21)]}
        return {"now_s": round(now / RATE, 3), "phrases": out, "horizon": horizon}

    def snapshot(self):
        st = self.status()
        return {"style": st["style"], "section": st["section"], "key": st["camelot"], "mode": st["mode"],
                "layers": [s_ for s_ in st["layers"] if s_ not in st["muted"]],
                "energy": st["energy"], "density": st["density"], "swing": st["swing"],
                "brightness": st["brightness"], "pattern_slots": sorted(self._slot_patterns)}

    def feedback(self, up):
        rec = self.prefs.record(self.snapshot(), up)
        self._log({"event": "feedback", "up": bool(up), "snapshot": rec, "t": self._elapsed()})

        def do():
            # the taste loop: liked sections come back more often, liked motifs return
            self.composer.form.taste = self.prefs.section_bias(self.style_name)
            if up:
                self.composer.melody.like()
        self._post(do)
        return rec

    # -- scenes -----------------------------------------------------------
    def scene_save(self, name):
        rec = self.scenes.save(name, scene_snapshot(self))
        self._log({"event": "scene_save", "name": name, "t": self._elapsed()})
        return rec

    def scene_load(self, name):
        from lib.gen.director import apply_intent
        sc = self.scenes.get(name)
        if sc is None:
            self.last_error = f"no scene {name!r}"
            return []
        if sc.get("style") and sc["style"] != self.style_name:
            self.set_style(sc["style"])
        done = apply_intent(self, scene_intent(sc, self.composer.style["slots"].keys()))
        self._director_log.append({"kind": "scene", "text": name, "say": "scene recalled", "done": done, "t": self._elapsed()})
        self._log({"event": "scene_load", "name": name, "done": done, "t": self._elapsed()})
        return done

    def scene_delete(self, name):
        self.scenes.delete(name)

    # -- director ---------------------------------------------------------
    def gesture(self, name):
        from lib.gen.director import apply_intent, gesture_intent
        intent = gesture_intent(name)
        if intent is None:
            self.last_error = f"unknown gesture {name!r}"
            return []
        done = apply_intent(self, intent)
        self._director_log.append({"kind": "gesture", "text": name, "say": intent.get("say", name), "done": done, "t": self._elapsed()})
        self._log({"event": "gesture", "name": name, "done": done, "t": self._elapsed()})
        return done

    def ask(self, text, transport=None):
        """Free text -> the LLM director -> Intent -> steering. Runs on a
        worker thread (seconds of latency); status shows busy/last reply."""
        from lib.gen.director import LLMDirector, apply_intent
        if self.director is None or transport is not None:
            self.director = LLMDirector(transport=transport)
        if self._director_busy:
            return False
        self._director_busy = True
        self._director_last = {"text": text, "t": self._elapsed()}

        def work():
            try:
                sandbox = None
                try:
                    from lib.gen.composer.strudel import open_engine
                    sandbox = open_engine()
                except Exception:
                    sandbox = None
                intent, warn, reply = self.director.intent_for(text, self.status(), slots=self.composer.style["slots"].keys(), sandbox=sandbox)
                if sandbox is not None:
                    sandbox.stop()
                done = apply_intent(self, intent)
                say = intent.get("say", "")
                self.director.history.append((text, say, done))
                self._director_last = {"text": text, "say": say, "done": done, "warn": warn, "error": "", "t": self._elapsed()}
                self._director_log.append({"kind": "ask", "text": text, "say": say, "done": done, "warn": warn, "t": self._elapsed()})
                self._log({"event": "ask", "text": text, "intent": intent, "warn": warn, "t": self._elapsed()})
            except Exception as e:  # noqa: BLE001
                self._director_last = {"text": text, "error": f"{type(e).__name__}: {e}", "t": self._elapsed()}
                self._log({"event": "ask_error", "text": text, "error": str(e), "t": self._elapsed()})
            finally:
                self._director_busy = False
        threading.Thread(target=work, daemon=True, name="gen-director").start()
        return True

    def reseed(self, seed=None):
        s = int(seed) if seed is not None else random.randrange(1, 10 ** 6)
        self.seed = s
        self._post(lambda: (self.composer.reseed(s), self._log({"event": "reseed", "seed": s, "t": self._elapsed()})))

    def set_set_length(self, seconds):
        self.set_length_s = float(seconds)
        self._post(lambda: self.composer.set_arc(self._arc_fn()))

    def request_end(self):
        self._post(lambda: (self.composer.request_end(), self._log({"event": "end_requested", "t": self._elapsed()})))

    # -- Strudel patterns -------------------------------------------------
    def set_pattern(self, code):
        """Evaluate Strudel code; from the next phrase its events replace
        the rule composer's. Bad code is reported in status, never fatal."""
        def do():
            from lib.gen.composer.strudel import StrudelSource, open_engine
            try:
                if self._strudel is None:
                    self._strudel = open_engine()
                src = self.composer.pattern_source
                if src is None or not isinstance(src, StrudelSource):
                    src = StrudelSource(self._strudel, self.composer.style["slots"].keys())
                src.load(code)
                self.composer.pattern_source = src
                self._pattern_code = code
                self._log({"event": "pattern", "code": code[:400], "t": self._elapsed()})
            except Exception as e:  # noqa: BLE001
                self.last_error = f"pattern: {type(e).__name__}: {e}"
        self._post(do)

    def clear_pattern(self):
        def do():
            self.composer.pattern_source = None
            self._pattern_code = None
            self.last_error = ""
            self._log({"event": "pattern_clear", "t": self._elapsed()})
        self._post(do)

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
            "lanes": self.rack.lane_values(), "lufs": round(float(self.rack.loud.lufs()), 1),
            "norm_db": round(self.rack.norm_db, 2), "stolen": stats.get("stolen", 0),
            "humanize": round(float(getattr(c, "humanize", 1.0)), 2),
            "automation": bool(getattr(c, "automation", True)),
            "timeline": self.timeline(),
            "script": ({"title": c.script.get("title"), "entry": self.composer.form.script_i, "n": len(c.script["sections"])}
                       if getattr(c, "script", None) and c.form.script else None),
            "hooks": dict(self.hooks.status(), source=getattr(c.melody, "source", "?"),
                          theme=(c.melody.theme.name or "(unnamed)") if c.melody.theme is not None else None),
            "render_errors": stats.get("render_errors", 0),
            "last_render_error": stats.get("last_render_error", ""),
            "motifs": len(c.melody.memory),
            "lead_s": round((self.rack.pending_until() - self.rack.clock) / RATE, 1),
            "log": list(self._log_tail)[-14:],
            "brightness": round(self.brightness, 3),
            "ramps": {k: {"to": r["to"], "bars_left": max(0, r["start_bar"] + r["bars"] - c.bar)} for k, r in self._ramps.items()},
            "section_requested": f.requested,
            "section_composed": (self._phrases[-1].section if self._phrases else None),   # already written, not yet heard
            "pattern_slots": sorted(self._slot_patterns),
            "slot_pattern_errors": {s_: src.error for s_, src in c.slot_patterns.items() if getattr(src, "error", "")},
            "gestures": _gesture_menu(),
            "director": {"available": (self.director.available if self.director is not None else _director_available()),
                         "mode": (self.director.mode if self.director is not None else None),
                         "busy": self._director_busy, "last": self._director_last,
                         "log": list(self._director_log)[-8:]},
            "taste": self.prefs.counts(),
            "scenes": self.scenes.listing(),
            "pattern": self._pattern_code,
            "pattern_error": (getattr(c.pattern_source, "error", "") if c.pattern_source else ""),
            "pattern_available": _strudel_available(),
            "pattern_engine": (type(self._strudel).__name__.replace("Strudel", "").lower()
                               if self._strudel is not None else None),
            "error": self.last_error,
        }

    def live_beat(self):
        """Ground-truth beat of what the room hears now (the composer's
        clock minus the engine's render lead). None when idle. `drive` is
        the phrase's actual rhythm: 1 with a kick playing, 0 in a break -
        a resting kick must not flash the room (the DJ's rule)."""
        if not self._running or self.composer is None or self.rack is None:
            return None
        heard = self._heard_clock()
        p = self._phrase_at(heard)
        c = self.composer
        beat = c.samples_per_beat
        spb = c.samples_per_bar
        pos = (heard - p.start) if p else heard
        layers = set(p.meta.get("layers", [])) if p else set()
        muted = set(self.muted)
        kick = "kick" in layers and "kick" not in muted
        bass = "bass" in layers and "bass" not in muted
        return {"bpm": float(c.bpm), "phase": float((pos % beat) / beat),
                "bar_phase": float((pos % spb) / spb),
                "phrase_phase": float((heard - p.start) / (p.end - p.start)) if p else 0.0,
                "bass_share": 0.5 if bass else (0.35 if kick else 0.15),
                "drive": 1.0 if kick else (0.4 if (layers - {"pad", "keys"}) else 0.0)}

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
