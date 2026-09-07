"""Composer: turns style + key + tempo + an arc into Phrases of NoteEvents,
four bars at a time, on an integer sample clock.

Deterministic under a seed. All musical decisions happen here; the synth
backends only render what they are told. Steering (energy bias, density,
key/tempo changes) takes effect at the next phrase boundary - the unit at
which a listener hears a change as intentional.

Order inside a phrase matters because the parts listen to each other:
kick first, then bass (avoids the kick), then the rest of the kit (hats
thin out under a busy bass), then pad / lead / keys (keys answer the
lead) / arp, then transition material the form knows about (risers into
a drop, impact and sweep on it) and the MIX AUTOMATION the section
implies (the high-pass climbing through a build, the low-pass closing a
break, sidechain depth and reverb per section) as "auto" events."""
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
# Micro-timing per slot, seconds (min, max) added to the grid position.
# The kick is the anchor and never moves; hats lag a touch (a lazy
# right hand), the bass sits just behind the kick, melodic layers float
# a few ms either side. Scaled by Composer.humanize. The style's "feel"
# adds a per-slot push on top.
HUMANIZE_S = {
    "kick": (0.0, 0.0), "snare": (-0.0015, 0.002), "hat": (0.0, 0.006), "ohat": (0.0, 0.005),
    "perc": (-0.004, 0.005), "bass": (0.0, 0.003), "keys": (-0.002, 0.007),
    "arp": (-0.002, 0.005), "lead": (-0.003, 0.009), "pad": (0.0, 0.0),
    "shaker": (-0.002, 0.006), "ride": (0.0, 0.004), "tom": (-0.002, 0.003), "rim": (-0.002, 0.003),
    "fx": (0.0, 0.0), "auto": (0.0, 0.0),
}
# Mix automation per section: lane -> (value, ramp in bars). "sweep" on
# the build's hp = climb from 20 Hz to ~500 Hz across the build, snapped
# open on the drop. A style may override per section in style["auto"].
AUTO = {
    "intro":  {"hp": (20.0, 0), "lp": (20000.0, 0), "duck": (0.3, 0), "verb": (1.2, 0), "delay_fb": (0.42, 0)},
    "groove": {"hp": (20.0, 0), "lp": (20000.0, 2), "duck": (0.45, 0), "verb": (1.0, 2), "delay_fb": (0.42, 0)},
    "build":  {"hp": "sweep", "lp": (20000.0, 0), "duck": (0.5, 0), "verb": (1.3, 2), "delay_fb": (0.5, 0)},
    "drop":   {"hp": (20.0, 0), "lp": (20000.0, 0), "duck": (0.6, 0), "verb": (0.8, 0), "delay_fb": (0.42, 0)},
    "break":  {"hp": (20.0, 0), "lp": (3500.0, 4), "duck": (0.15, 0), "verb": (1.7, 2), "delay_fb": (0.55, 0)},
    "outro":  {"hp": (20.0, 0), "lp": (900.0, 16), "duck": (0.3, 0), "verb": (1.4, 4), "delay_fb": (0.5, 0)},
    "flow":   {"hp": (20.0, 0), "lp": (20000.0, 4), "duck": (0.2, 0), "verb": (1.5, 2), "delay_fb": (0.5, 0)},
    "swell":  {"hp": (20.0, 0), "lp": (20000.0, 2), "duck": (0.3, 0), "verb": (1.2, 2), "delay_fb": (0.45, 0)},
    "calm":   {"hp": (20.0, 0), "lp": (5000.0, 8), "duck": (0.1, 0), "verb": (1.8, 4), "delay_fb": (0.6, 0)},
}


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
        self.melody = Melody(self.style, self.key, self.rng, harmony=self.harmony)
        self.drums = Drums(self.style, self.rng)
        self.clock = 0              # sample of the next phrase's downbeat
        self.bar = 0
        # live steering (applied at phrase boundaries)
        self.energy_bias = 0.0      # -0.5 .. +0.5
        self.density = 1.0          # scales probabilistic layers
        self.swing = self.style["swing"]
        self.feel = self.style.get("feel", {})
        self.muted = set()
        self._pending_key = None
        self._pending_bpm = None
        self.log = []
        # Optional external note source (a Strudel pattern): when set, its
        # events replace the rule-generated ones; form/harmony/energy keep
        # running and are handed to it as context.
        self.pattern_source = None
        # Per-slot pattern overrides: {slot: source}. Those slots take their
        # notes from the pattern; every other slot stays with the rules.
        self.slot_patterns = {}
        # Timbre lever the director can move: multiplies each pitched
        # patch's filter cutoff (0.4 dark .. 1.6 bright).
        self.brightness = 1.0
        # Humanization amount (0 = machine grid, 1 = the default feel):
        # per-slot micro-timing, chord strums, bass slides, groove pushes.
        self.humanize = 1.0
        # Mix automation on/off (the director can freeze the lanes).
        self.automation = True
        self._riser_for = None      # drop bar a riser has already been written for
        self._auto_state = {}       # lane -> last target written
        self._last_section = None
        self.script = None          # the SongScript being followed (lib/gen/script.py)
        self._script_entry = None   # the entry whose levers are applied
        self._script_lanes = {}

    # -- scripted songs --------------------------------------------------------
    def load_script(self, script: dict):
        """Follow a SongScript from the next phrase: sections and lengths
        from the script, plus each section's levers (energy, density,
        brightness, swing, layers, chords, lanes, key, bpm, hook)."""
        from lib.gen.script import normalize
        s = normalize(script)
        self.script = s
        self.humanize = float(s.get("humanize", 1.0))
        if s.get("bpm"):
            self.bpm = float(s["bpm"])
        self.form.set_script(s["sections"], end=bool(s.get("end", True)))
        self._script_entry = None
        self._script_lanes = {}
        # the song's MATERIAL into the generator's mechanisms
        if s.get("motifs"):
            from lib.gen.composer.melody import Motif
            seeded = []
            for m in s["motifs"]:
                mo = Motif(m["steps"], m["degrees"], self.bar, contour=m.get("contour", "flat"), name=m.get("name", "song"))
                mo.liked = max(0, int(m.get("count", 1)) - 1)       # recurrence = how much the song liked it
                self.melody.memory.add(mo)
                seeded.append(mo)
            if seeded and self.melody.theme is None:
                self.melody.theme = seeded[0]
            self.melody.hook_provider = lambda rng, _s=seeded: (lambda m: {"steps": m.steps, "degrees": m.degrees, "contour": m.contour, "name": m.name})(
                rng.choices(_s, weights=[1 + m.liked for m in _s])[0])
        if s.get("bass_cells"):
            self.melody.bass_library = [dict(c) for c in s["bass_cells"]]

    def _apply_script_entry(self):
        e = self.form.script_entry
        if e is None:
            if self._script_entry is not None:           # script over: release the overrides
                self._script_entry = None
                self.harmony.override = None
                self.melody.bass_override = None
                self.drums.override = None
                self._script_lanes = {}
            return
        if e is self._script_entry:
            return
        self._script_entry = e
        if e.get("bpm"):
            self.bpm = float(e["bpm"])
        if e.get("key"):
            self.key = parse_key(str(e["key"]))
            self.harmony.modulate(self.key)
            self.melody.key = self.key
        if e.get("density") is not None:
            self.density = float(e["density"])
        if e.get("brightness") is not None:
            self.brightness = float(e["brightness"])
        if e.get("swing") is not None:
            self.swing = float(e["swing"])
        self.melody.bass_override = dict(e["bass"]) if e.get("bass") else None
        self._script_lanes = dict(e.get("lanes") or {})
        self._apply_phrase_overrides()
        h = e.get("hook")
        if h and h.get("steps") and h.get("degrees"):
            from lib.gen.composer.melody import Motif
            m = Motif(h["steps"], h["degrees"], self.bar, contour=h.get("contour", "flat"), name=str(h.get("name", "script")))
            if h.get("answer"):
                m.answer = (list(h["answer"]["steps"]), list(h["answer"]["degrees"]))
            self.melody.theme = m
            self.melody.memory.add(m)
            self.melody.hook_provider = lambda rng, _m=m: {"steps": _m.steps, "degrees": _m.degrees, "contour": _m.contour, "name": _m.name}

    def _apply_phrase_overrides(self):
        """The scripted section's per-phrase material: this phrase's four
        chords (the section's chord list runs bar by bar and cycles) and
        this phrase's drum template (drums_phrases, else the section's)."""
        e = self._script_entry
        if e is None:
            return
        p0 = int(self.form.bars_in)
        ch = e.get("chords")
        self.harmony.override = [ch[(p0 + b) % len(ch)] for b in range(PHRASE_BARS)] if ch else None
        tpl = None
        dp = e.get("drums_phrases")
        if dp:
            tpl = dp[(p0 // PHRASE_BARS) % len(dp)]
        elif e.get("drums"):
            tpl = e["drums"]
        if tpl:
            ov = {k: [(int(st), float(v)) for st, v in (tpl.get(k) or [])] for k in tpl if k != "fill" and tpl.get(k) is not None}
            if tpl.get("fill"):
                ov["fill"] = {k: [(int(st), float(v)) for st, v in hits] for k, hits in tpl["fill"].items()}
            self.drums.override = ov or None
        else:
            self.drums.override = None
        db = e.get("drums_bars")
        if db:
            self.drums.override_bars = [{k: [(int(st), float(v)) for st, v in hits] for k, hits in db[(p0 + b) % len(db)].items()}
                                        for b in range(PHRASE_BARS)]
        else:
            self.drums.override_bars = None
        # steering relative to the section's own levers: 1.0 = play the song's beat as identified
        e_en = float(e.get("energy") if e.get("energy") is not None else 0.6)
        e_de = float(e.get("density") if e.get("density") is not None else 1.0)
        self.drums.steer = min(1.5, max(0.1, (max(0.0, self.form.energy() + self.energy_bias) / max(e_en, 0.05)) * (self.density / max(e_de, 0.05))))

    def _dyn(self, bar_in_section: int) -> float:
        """The scripted section's dynamics (dB relative to its mean) at a
        bar, scaled down by the fidelity: the source loops carry their own."""
        e = self._script_entry
        if e is None:
            return 0.0
        trim = float(e.get("trim_db") or 0.0)             # level calibration (script.render's second pass)
        if not e.get("dyn"):
            return trim
        d = e["dyn"]
        k = 1.0 - float((self.script or {}).get("fidelity", 0.0) or 0.0)
        return trim + k * float(d[bar_in_section % len(d)])

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

    def _step_to_sample(self, bar0_sample: int, step_abs: int, slot: str | None = None) -> int:
        S = self.style["steps_per_bar"]
        bar, step = divmod(step_abs, S)
        spb = self.samples_per_bar
        sps = spb / S
        t = bar * spb + step * sps
        swing = self.swing
        if slot is not None:
            swing *= float(self.feel.get(slot, {}).get("swing", 1.0))
        if swing > 0 and step % 2 == 1:        # delay the off-16ths
            t += swing * sps * 2.0 * 0.5
        return int(round(bar0_sample + t))

    # -- the phrase -----------------------------------------------------------
    def next_phrase(self) -> Phrase:
        if self._pending_key is not None:
            self.key = self._pending_key
            self.harmony.modulate(self.key)
            self.melody.key = self.key
            self.melody.retire_theme()      # a new key is a new movement: new hook
            self._pending_key = None
        if self._pending_bpm is not None:
            self.bpm = self._pending_bpm
            self._pending_bpm = None
        self._apply_script_entry()
        self._apply_phrase_overrides()
        nb = PHRASE_BARS
        S = self.style["steps_per_bar"]
        section = self.form.section
        # a louder phrase of the source is also a busier one: +6 dB ~ +0.12 energy
        dyn_bias = 0.02 * sum(self._dyn(self.form.bars_in + b) for b in range(nb)) / nb
        energy = max(0.0, min(1.0, self.form.energy() + self.energy_bias + dyn_bias))
        layers = self.form.layers() - self.muted
        chords = self.harmony.next_phrase(nb, section)
        start = self.clock
        spb = self.samples_per_bar
        sps = spb / S
        last_of_section = self.form.bars_left <= nb
        section_start = self.form.bars_in == 0 or section != self._last_section
        self._last_section = section
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
            self._automation(ev, section, section_start, start, spb, nb)
            return self._finish_phrase(ev, nb, start, spb, section, energy, chords, layers, None)

        phrase_end = start + nb * spb
        hum = float(self.humanize)

        def note(step_abs, slot, pitch, vel, dur_steps, offset_s=0.0, **params):
            at = self._step_to_sample(start, step_abs, slot)
            lo, hi = HUMANIZE_S.get(slot, (0.0, 0.0))
            push = float(self.feel.get(slot, {}).get("push", 0.0))
            at += int((self.rng.uniform(lo, hi) * hum + (offset_s + push) * hum) * RATE)
            at = max(start, min(at, phrase_end - 1))
            dur = max(int(dur_steps * sps) - 1, int(0.02 * RATE))
            if slot not in DRUM_SLOTS and slot != "fx" and abs(self.brightness - 1.0) > 1e-6 and "cutoff" not in params:
                base = self.style["slots"].get(slot, {}).get("cutoff")
                if base:
                    params = dict(params, cutoff=float(base) * float(self.brightness))
            ev.append(NoteEvent(at, slot, float(pitch), float(min(1.0, vel)), dur, params))

        halftime = section in self.style.get("halftime_in", ())
        double = section == "build"
        phrase_id = self.bar // nb
        base_ctx = {"section": section, "phrase": phrase_id, "halftime": halftime, "double": double,
                    "poly": bool(self.style.get("perc_poly", False))}
        # kick first: everything else is placed around it
        kick_steps = [set() for _ in range(nb)]
        if "kick" in layers:
            for b in range(nb):
                ctx = dict(base_ctx, bar=self.bar + b)
                for step, vel in self.drums.bar("kick", energy, b, nb, last_of_section, ctx):
                    if step < S:
                        note(b * S + step, "kick", 36.0, vel, 1)
                        kick_steps[b].add(step)
        # bass: avoids the kick, slides between its own notes
        bass_hits = [0] * nb
        if "bass" in layers:
            prev_b = None            # (step the previous note ends, its pitch)
            for b in range(nb):
                notes_b = self.melody.bass_bar(chords[b], energy, b, kick_steps[b])
                bass_hits[b] = len(notes_b)
                for step, midi, vel, gate in notes_b:
                    extra = {}
                    s_abs = b * S + step
                    if (prev_b is not None and hum > 0 and prev_b[1] != midi
                            and s_abs - prev_b[0] <= 1 and self.rng.random() < 0.3 * hum):
                        extra["glide_from"] = prev_b[1]      # 303-style slide into this note
                    note(s_abs, "bass", midi, vel, gate, **extra)
                    prev_b = (s_abs + gate, midi)
        # the rest of the kit, aware of kick and bass
        for slot in DRUM_SLOTS:
            if slot == "kick" or slot not in layers:
                continue
            for b in range(nb):
                ctx = dict(base_ctx, bar=self.bar + b, bass_hits=bass_hits[b], kick_steps=kick_steps[b])
                for hit in self.drums.bar(slot, energy * self.density, b, nb, last_of_section, ctx):
                    step, vel = hit[0], hit[1]
                    pitch = hit[2] if len(hit) > 2 else 36.0
                    if step < S:
                        note(b * S + step, slot, pitch, vel, 1)
        # pad: one voicing per chord change, held to the next change
        if "pad" in layers:
            prev = None
            for b in range(nb):
                if prev is not None and chords[b][0] == prev[0] and chords[b][2] == prev[4]:
                    prev = (prev[0], prev[1] + 1, prev[2], prev[3], prev[4])
                    continue
                if prev is not None:
                    for j, m in enumerate(prev[2]):
                        note(prev[3] * S, "pad", m, 0.55 + 0.1 * self.rng.random(), prev[1] * S, offset_s=0.02 * j)
                voicing = self.melody.pad_bar(chords[b], energy)
                pad_patch = self.style["slots"].get("pad", {})
                if pad_patch.get("drone"):
                    # a song's own sustained texture: one note on the chord root, in the sample's own register
                    base = int(pad_patch.get("samples", [{}])[0].get("base_midi", 60)) if pad_patch.get("samples") else 60
                    m = self.key.degree_midi(self.harmony.root_degree(chords[b]), 3)
                    while m - base > 6:
                        m -= 12
                    while base - m > 6:
                        m += 12
                    voicing = [m]
                prev = (chords[b][0], 1, voicing, b, chords[b][2])
            if prev is not None:
                for j, m in enumerate(prev[2]):
                    note(prev[3] * S, "pad", m, 0.55 + 0.1 * self.rng.random(), prev[1] * S, offset_s=0.02 * j, sustain=True)
        # lead motif (and the theme: made in the build, stated on the drop)
        op = None
        lead_steps = [set() for _ in range(nb)]
        if "lead" in layers:
            theme = None
            if section == "build":
                theme = "make"
            elif section == "drop" and self.form.bars_in == 0:
                theme = "state"
            lead_ctx = {"section": section, "last_of_section": last_of_section and section not in ("build",),
                        "climax": section == "build" and last_of_section}
            notes, op = self.melody.lead_phrase(chords, energy * self.density, nb, theme, lead_ctx)
            for step_abs, midi, vel, gate in notes:
                note(step_abs, "lead", midi, vel, gate)
                b, s = divmod(step_abs, S)
                if b < nb:
                    lead_steps[b].add(s)
        # keys stabs: answer the lead
        if "keys" in layers:
            for b in range(nb):
                for step, chord, vel, gate in self.melody.keys_bar(chords[b], energy * self.density, lead_steps[b]):
                    strum = self.rng.uniform(0.004, 0.011)
                    for j, m in enumerate(chord):
                        note(b * S + step, "keys", m, vel * (1.0 - 0.05 * j), gate, offset_s=strum * j)
        # arp
        if "arp" in layers:
            for b in range(nb):
                for step, midi, vel, gate in self.melody.arp_bar(chords[b], energy * self.density):
                    note(b * S + step, "arp", midi, vel, gate)
        # transitions the form knows about
        if "fx" in self.style["slots"] and "fx" not in self.muted:
            drop_bar = self.form.upcoming_drop_bar()
            if drop_bar is not None:
                to_drop = drop_bar - self.bar
                if 0 < to_drop <= 8 and self._riser_for != drop_bar:
                    self._riser_for = drop_bar
                    note(0, "fx", 45.0, 0.8, to_drop * S, kind="riser")
                if to_drop == nb:
                    note((nb - 1) * S, "fx", 60.0, 0.7, S, kind="revcym")
            if section == "drop" and self.form.bars_in == 0:
                note(0, "fx", 36.0, 1.0, 2 * S, kind="impact")
                note(0, "fx", 60.0, 0.6, 2 * S, kind="sweep")
        self._automation(ev, section, section_start, start, spb, nb)
        # the source song's own loops for this section (fidelity > 0): one hit per phrase per stem
        loops_on = set()
        if self.script and self._script_entry and self._script_entry.get("loops") and self.script.get("fidelity", 0.0) > 0.0:
            fid = float(self.script["fidelity"])
            rate = (self.bpm / float(self.script["bpm_src"])) if self.script.get("bpm_src") else 1.0
            stems = self._script_entry["loops"]
            order = ["drums", "bass", "other", "vocals"]
            n_on = 4 if fid >= 0.99 else (3 if fid >= 0.75 else (2 if fid >= 0.4 else 1))
            for name in order[:n_on]:
                path = stems.get(name)
                if not path:
                    continue
                slot = {"drums": "loop_drums", "bass": "loop_bass", "other": "loop_other", "vocals": "loop_vox"}[name]
                if slot in self.muted:
                    continue
                loops_on.add(name)
                dur_steps = nb * S / rate
                note(0, slot, 60.0, 1.0, dur_steps, file=path, rate=rate)
            # the generator steps back where a loop plays
            if "drums" in loops_on:
                ev = [e for e in ev if e.slot not in DRUM_SLOTS]
            if "bass" in loops_on:
                ev = [e for e in ev if e.slot != "bass"]
            if "other" in loops_on:
                ev = [e for e in ev if e.slot not in ("pad", "keys", "arp", "lead")]
        # (the transcribed lines are evidence only: the lead develops the song's motifs and the bass
        #  draws from its cell library - both generated, both steerable - through its note samples)
        # the source song's own vocal phrases, where they were
        if self.script and self.script.get("vocals") and "vox" not in self.muted and "vocals" not in loops_on:
            rate = (self.bpm / float(self.script["bpm_src"])) if self.script.get("bpm_src") else 1.0
            for v in self.script["vocals"]:
                b = float(v["bar"])
                if self.bar <= b < self.bar + nb:
                    dur_steps = max(1.0, float(v.get("seconds", 1.0)) / rate * RATE / sps)
                    note((b - self.bar) * S, "vox", 60.0, 0.9, dur_steps, file=v["file"], rate=rate)

        if self.slot_patterns:
            ctx = {"energy": round(energy, 3), "section": section, "bar": self.bar,
                   "key": self.key.camelot, "bpm": round(self.bpm, 2), "phrase": self.bar // nb,
                   "chords": [c[1] for c in chords]}
            ev = [e for e in ev if e.slot not in self.slot_patterns]
            for slot, src in list(self.slot_patterns.items()):
                try:
                    ev += [e for e in src.events(self.bar, nb, start, spb, ctx) if e.slot == slot]
                    src.error = ""
                except Exception as e:  # noqa: BLE001
                    src.error = f"{type(e).__name__}: {e}"
        return self._finish_phrase(ev, nb, start, spb, section, energy, chords, layers, op)

    def _automation(self, ev, section, section_start, start, spb, nb):
        """Write the section's mix automation as "auto" events (lane, to,
        ramp). Only lanes whose target changed are written, so a long
        groove does not repeat itself; the build's high-pass climbs a
        step every phrase."""
        if not self.automation:
            return
        prog = dict(AUTO.get(section, {}))
        prog.update(self.style.get("auto", {}).get(section, {}))
        if self._script_lanes:
            prog.update({k: (float(v), 1) for k, v in self._script_lanes.items()})
        for lane, spec in prog.items():
            if spec == "sweep":
                total = max(1, self.form.bars_in + self.form.bars_left)
                p_end = min(1.0, (self.form.bars_in + nb) / total)
                to, ramp_bars = 20.0 * (25.0 ** p_end), nb
                if self._auto_state.get(lane) == round(to, 1):
                    continue
                self._auto_state[lane] = round(to, 1)
            else:
                to, ramp_bars = spec
                if not section_start and self._auto_state.get(lane) == to:
                    continue
                if self._auto_state.get(lane) == to:
                    continue
                self._auto_state[lane] = to
            ev.append(NoteEvent(start, "auto", 0.0, 0.0, int(ramp_bars * spb), {"lane": lane, "to": float(to)}))
        # the scripted section's dynamics: the gain lane bar by bar (a short ramp into each bar)
        e = self._script_entry
        if e is not None and (e.get("dyn") or e.get("trim_db")):
            for b in range(nb):
                to = round(10.0 ** (self._dyn(self.form.bars_in + b) / 20.0), 4)
                ev.append(NoteEvent(int(start + b * spb), "auto", 0.0, 0.0, int(spb / 8), {"lane": "gain", "to": float(to)}))
            self._auto_state["gain"] = to
        elif self._auto_state.get("gain", 1.0) != 1.0:
            ev.append(NoteEvent(start, "auto", 0.0, 0.0, int(spb / 2), {"lane": "gain", "to": 1.0}))
            self._auto_state["gain"] = 1.0

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
