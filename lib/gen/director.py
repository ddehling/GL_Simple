"""The director: interaction ABOVE code. Three inputs, one Intent:

  gestures   a small vocabulary with musical meaning (darker, open it up,
             strip to drums, build to a drop...) - each a fixed Intent;
  language   free text -> an LLM (Claude, via the `claude` CLI or the
             anthropic SDK - same transports as the DJ planner's copilot)
             -> an Intent JSON, validated and sandboxed before it is applied;
  taste      thumbs up/down on phrases (lib/gen/feedback.py); 'more like
             this' is a gesture that pulls parameters toward liked ground.

An Intent is a plain dict; every field is optional:
  say        str      one line back to the operator
  set        {energy_bias, density, swing, brightness, bpm, master, key}  absolute
  nudge      {energy_bias, density, swing, brightness, bpm}               relative
  ramp       {param: {"to": x, "bars": n}}   linear over n bars (phrase-quantised)
  section    "build" | "drop" | "break" | "groove" | "outro" | ...        next section
  hold       bool     stay in the current section
  reseed     bool
  end        bool     play the outro and stop
  layers     {"mute": [slots], "unmute": [slots]}
  patterns   {slot: "<strudel code>"}   notes for those slots from a pattern
  pattern    "<strudel code>" | ""      whole-rack pattern ("" clears)
  like       bool     record taste (true = more like this)

Everything lands at phrase boundaries through GenSystem's steering queue,
so a gesture, a sentence and a thumbs-up all show up in the phrase log as
auditable changes; the autonomous composer keeps running underneath."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time

from lib.gen.composer.styles import STYLES
from lib.gen.events import SLOTS

MODEL = os.environ.get("GEN_DIRECTOR_MODEL", "claude-opus-5")
PITCHED = ("bass", "lead", "pad", "arp", "keys")
DRUMS = ("kick", "snare", "hat", "ohat", "perc")

# -- the gesture vocabulary --------------------------------------------------
GESTURES = {
    "darker":        {"label": "darker",         "nudge": {"brightness": -0.25, "energy_bias": -0.08}},
    "brighter":      {"label": "brighter",       "nudge": {"brightness": 0.25, "energy_bias": 0.05}},
    "open_up":       {"label": "open it up",     "layers": {"unmute": list(SLOTS)}, "nudge": {"density": 0.2, "energy_bias": 0.1}},
    "strip_drums":   {"label": "strip to drums", "layers": {"mute": list(PITCHED)}, "hold": True},
    "melody":        {"label": "bring a melody", "layers": {"unmute": ["lead", "arp"]}, "nudge": {"density": 0.1}},
    "no_melody":     {"label": "lose the melody", "layers": {"mute": ["lead", "arp"]}},
    "bass_in":       {"label": "bass in",        "layers": {"unmute": ["bass"]}},
    "sparser":       {"label": "sparser",        "nudge": {"density": -0.25}},
    "denser":        {"label": "denser",         "nudge": {"density": 0.25}},
    "swingier":      {"label": "more swing",     "nudge": {"swing": 0.06}},
    "straighter":    {"label": "straighter",     "nudge": {"swing": -0.06}},
    "slower":        {"label": "slower",         "nudge": {"bpm": -4}},
    "faster":        {"label": "faster",         "nudge": {"bpm": 4}},
    "build":         {"label": "build to a drop", "section": "build", "hold": False},
    "breakdown":     {"label": "breakdown",      "section": "break", "hold": False},
    "groove":        {"label": "back to groove", "section": "groove", "hold": False},
    "modulate_up":   {"label": "modulate up",    "key": "up"},
    "modulate_down": {"label": "modulate down",  "key": "down"},
    "keep":          {"label": "keep this going", "hold": True},
    "release":       {"label": "let it move",    "hold": False},
    "wind_down":     {"label": "wind down",      "ramp": {"energy_bias": {"to": -0.35, "bars": 32}, "density": {"to": 0.5, "bars": 32}}, "section": "break"},
    "more_like_this": {"label": "more like this", "like": True},
    "reseed":        {"label": "new ideas",      "reseed": True},
}

CLAMP = {"energy_bias": (-0.5, 0.5), "density": (0.0, 1.5), "swing": (0.0, 0.33),
         "brightness": (0.4, 1.6), "bpm": (50.0, 180.0), "master": (0.0, 1.0)}


def _clampf(k, v):
    lo, hi = CLAMP[k]
    return max(lo, min(hi, float(v)))


def validate_intent(raw, slots=SLOTS, sandbox=None):
    """Normalise an Intent from any source (gesture table, LLM JSON, tests).
    Unknown keys are dropped, numbers clamped, sections/slots checked,
    patterns evaluated in `sandbox` (a Strudel engine) when given. Returns
    (intent, warnings); never raises on bad content."""
    warn = []
    out = {}
    if not isinstance(raw, dict):
        return {}, ["intent is not an object"]
    if isinstance(raw.get("say"), str):
        out["say"] = raw["say"].strip()[:300]
    for group in ("set", "nudge"):
        g = raw.get(group)
        if isinstance(g, dict):
            clean = {}
            for k, v in g.items():
                if k == "key" and group == "set" and isinstance(v, str):
                    clean["key"] = v.strip().upper()
                elif k in CLAMP:
                    try:
                        clean[k] = float(v) if group == "nudge" else _clampf(k, v)
                    except (TypeError, ValueError):
                        warn.append(f"{group}.{k}: not a number")
                else:
                    warn.append(f"{group}.{k}: unknown parameter")
            if clean:
                out[group] = clean
    if isinstance(raw.get("key"), str) and raw["key"] in ("up", "down"):
        out["key"] = raw["key"]
    r = raw.get("ramp")
    if isinstance(r, dict):
        ramp = {}
        for k, v in r.items():
            if k in CLAMP and isinstance(v, dict):
                try:
                    ramp[k] = {"to": _clampf(k, v.get("to")), "bars": max(4, min(256, int(v.get("bars", 16))))}
                except (TypeError, ValueError):
                    warn.append(f"ramp.{k}: bad")
        if ramp:
            out["ramp"] = ramp
    sec = raw.get("section")
    if isinstance(sec, str):
        if sec in {n for st in STYLES.values() for n in st["sections"]}:
            out["section"] = sec
        else:
            warn.append(f"section {sec!r} unknown")
    for b in ("hold", "reseed", "end", "like"):
        if b in raw and raw[b] is not None:
            out[b] = bool(raw[b])
    lay = raw.get("layers")
    if isinstance(lay, dict):
        clean = {}
        for k in ("mute", "unmute"):
            v = lay.get(k)
            if isinstance(v, (list, tuple)):
                good = [s for s in v if s in slots]
                bad = [s for s in v if s not in slots]
                if bad:
                    warn.append(f"layers.{k}: unknown slots {bad}")
                if good:
                    clean[k] = good
        if clean:
            out["layers"] = clean
    pats = raw.get("patterns")
    if isinstance(pats, dict):
        clean = {}
        for slot, code in pats.items():
            if slot not in slots:
                warn.append(f"patterns.{slot}: unknown slot")
                continue
            if not isinstance(code, str) or not code.strip():
                continue
            code = code.strip()[:8000]
            if sandbox is not None:
                ok, why = _sandbox_check(sandbox, code, slot)
                if not ok:
                    warn.append(f"patterns.{slot}: {why}")
                    continue
            clean[slot] = code
        if clean:
            out["patterns"] = clean
    if "pattern" in raw and (isinstance(raw["pattern"], str)):
        code = raw["pattern"].strip()[:20000]
        if code and sandbox is not None:
            ok, why = _sandbox_check(sandbox, code, None)
            if not ok:
                warn.append(f"pattern: {why}")
                code = None
        if code is not None:
            out["pattern"] = code
    return out, warn


def _sandbox_check(engine, code, slot):
    """Evaluate in a scratch engine and confirm it produces events for the
    slot (or any slot) inside one bar. The live rack never sees bad code."""
    try:
        engine.eval(code)
        haps = engine.query(0, 4, {"energy": 0.6})
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"
    if not haps:
        return False, "pattern produced no events"
    if slot is not None:
        from lib.gen.composer.strudel import StrudelSource
        hits = [h for h in haps if StrudelSource._slot(h.get("v") or {}) == slot]
        if not hits:
            return False, f"pattern produced no events for slot {slot!r} (use .s(\"{slot}\"))"
    if len(haps) > 4 * 64:
        return False, "pattern too dense (> 64 events per bar)"
    return True, ""


def gesture_intent(name):
    g = GESTURES.get(name)
    if g is None:
        return None
    intent = {k: v for k, v in g.items() if k != "label"}
    intent.setdefault("say", g["label"])
    return intent


# -- applying an intent to a live GenSystem -----------------------------------
def apply_intent(system, intent):
    """Translate an Intent into GenSystem steering calls (all queued to the
    conductor, applied at phrase boundaries). Returns a list of what was
    done, for the log and the page."""
    done = []
    st = system.status()
    cur = {"energy_bias": st.get("energy_bias", 0.0), "density": st.get("density", 1.0),
           "swing": st.get("swing", 0.0), "brightness": st.get("brightness", 1.0),
           "bpm": st.get("bpm", 120.0)}
    setters = {"energy_bias": system.set_energy_bias, "density": system.set_density,
               "swing": system.set_swing, "brightness": system.set_brightness,
               "bpm": system.set_bpm, "master": system.set_master}
    for k, v in (intent.get("set") or {}).items():
        if k == "key":
            system.set_key(v); done.append(f"key {v}")
        elif k in setters:
            setters[k](v); done.append(f"{k} {v:.2f}")
    for k, dv in (intent.get("nudge") or {}).items():
        if k in setters and k in cur:
            nv = _clampf(k, cur[k] + dv)
            setters[k](nv); done.append(f"{k} {cur[k]:.2f}->{nv:.2f}")
    if intent.get("key") in ("up", "down"):
        k = system.composer.key
        nk = k.relative(7 if intent["key"] == "up" else -7)
        system.set_key(nk); done.append(f"key {k.camelot}->{nk.camelot}")
    for k, spec in (intent.get("ramp") or {}).items():
        system.add_ramp(k, spec["to"], spec["bars"]); done.append(f"ramp {k}->{spec['to']:.2f} over {spec['bars']} bars")
    if intent.get("section"):
        system.request_section(intent["section"]); done.append(f"section {intent['section']}")
    if "hold" in intent:
        system.set_hold(intent["hold"]); done.append("hold" if intent["hold"] else "release")
    for s in (intent.get("layers") or {}).get("mute", []):
        system.set_mute(s, True); done.append(f"mute {s}")
    for s in (intent.get("layers") or {}).get("unmute", []):
        system.set_mute(s, False); done.append(f"unmute {s}")
    for slot, code in (intent.get("patterns") or {}).items():
        system.set_slot_pattern(slot, code); done.append(f"pattern for {slot}")
    if "pattern" in intent:
        if intent["pattern"]:
            system.set_pattern(intent["pattern"]); done.append("whole-rack pattern")
        else:
            system.clear_pattern(); done.append("pattern cleared")
    if intent.get("reseed"):
        system.reseed(); done.append("reseed")
    if intent.get("like") is not None:
        system.feedback(intent["like"]); done.append("liked" if intent["like"] else "disliked")
        if intent["like"]:
            for k, dv in system.prefs.nudges(system.style_name, cur).items():
                if k in setters:
                    setters[k](_clampf(k, cur[k] + dv)); done.append(f"{k} toward liked")
    if intent.get("end"):
        system.request_end(); done.append("end after outro")
    return done


# -- the language director -----------------------------------------------------
SYSTEM_PROMPT = """You are the director of an autonomous generative music system playing live in a club-style light show. The operator speaks to you in plain language; you answer with ONE JSON object (an Intent) and nothing else - no prose outside the JSON.

The music is composed by rules four bars at a time: a section state machine (intro, groove, build, drop, break, outro), chord loops in a key, Euclidean drums, a bass riff cell, a lead motif with memory, pads, keys stabs, arps. It runs for hours on an energy arc. You steer it; you do not replace it. Changes land at the next phrase boundary (4 bars).

Intent fields (all optional):
  "say": one short line back to the operator (what you did / why).
  "set": absolute values: energy_bias -0.5..0.5, density 0..1.5, swing 0..0.33, brightness 0.4..1.6 (filter cutoff multiplier), bpm 50..180, master 0..1, key (Camelot like "9A").
  "nudge": relative deltas for the same numeric parameters (prefer nudges for "a bit more/less").
  "ramp": {"param": {"to": value, "bars": n}} for "over the next N minutes/bars" (1 bar ~ 2 s at 124 bpm).
  "section": request the next section: "build" (then a drop follows), "drop", "break", "groove", "outro".
  "hold": true to stay in the current section, false to let it move.
  "layers": {"mute": [...], "unmute": [...]} over slots: kick snare hat ohat perc bass lead pad arp keys.
  "patterns": {"slot": "<Strudel code>"} to write NEW material for a slot when the operator asks for a specific figure (an arp, a riff, a hat pattern). One cycle = one bar. Use .s("slot") so it lands in that slot; note("...") with .scale("<root><octave>:<mode>") for pitched slots, e.g. note("0 2 4 7").scale("A3:minor").s("arp"). Globals you may use inside patterns: energy (0..1 signal), bar, bpm. Keep patterns musical and within 1-16 events per bar per slot.
  "pattern": whole-rack Strudel code replacing everything (rare); "" clears it.
  "reseed": true for "new ideas"; "end": true to finish after an outro; "like": true/false to record taste.

Guidance: small moves, musically named. "darker" = brightness down (and a little energy); "warmer" = brightness down slightly, density unchanged; "hypnotic" = hold + sparser + straighter; "lift" = build then drop; "strip it back" = mute pitched layers; "half-time feel" = swap in a slower kick pattern via patterns.kick rather than halving bpm. Never exceed bpm changes of 8 unless asked for a tempo. Prefer one clear action per request; combine only when the operator asks for a sequence.
"""


def find_claude_exe():
    exe = shutil.which("claude")
    if exe:
        return exe
    home = os.path.expanduser("~")
    for c in (os.path.join(home, ".local", "bin", "claude"), "/usr/local/bin/claude",
              os.path.join(home, "AppData", "Roaming", "npm", "claude.cmd")):
        if os.path.exists(c):
            return c
    return None


def _extract_json(text):
    """The first balanced {...} object in `text` (models sometimes wrap it)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        text = m.group(1)
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in the reply")


class LLMDirector:
    """Text -> Intent through Claude. `transport` may be injected (tests,
    other models): a callable (system, prompt) -> reply text."""

    def __init__(self, transport=None, model=MODEL):
        self.model = model
        self._transport = transport
        self._exe = None
        self._client = None
        self.mode = "injected" if transport else self._detect()
        self.history = []            # (operator text, say, done)

    def _detect(self):
        self._exe = find_claude_exe()
        if self._exe:
            return "cli"
        try:
            import anthropic
            self._client = anthropic.Anthropic()      # env / ant profile
            return "sdk"
        except Exception:
            return "unavailable"

    @property
    def available(self):
        return self.mode != "unavailable"

    def _call_cli(self, system, prompt):
        import tempfile
        sf = tempfile.NamedTemporaryFile(mode="w", suffix=".sys.txt", delete=False, encoding="utf-8")
        sf.write(system); sf.close()
        try:
            cmd = [self._exe, "--no-session-persistence", "--model", self.model,
                   "--system-prompt-file", sf.name, "--output-format", "text", "-p"]
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=120)
            if r.returncode != 0:
                raise RuntimeError((r.stderr or r.stdout or "claude failed").strip()[:300])
            return (r.stdout or "").strip()
        finally:
            try:
                os.unlink(sf.name)
            except OSError:
                pass

    def _call_sdk(self, system, prompt):
        response = self._client.messages.create(
            model=self.model, max_tokens=4000,
            thinking={"type": "adaptive"}, output_config={"effort": "low"},
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}])
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined")
        return "\n".join(b.text for b in response.content if getattr(b, "type", "") == "text")

    def call(self, system, prompt):
        if self._transport:
            return self._transport(system, prompt)
        if self.mode == "cli":
            return self._call_cli(system, prompt)
        if self.mode == "sdk":
            return self._call_sdk(system, prompt)
        raise RuntimeError("no director transport: install Claude Code (`claude`) or `pip install anthropic` + ANTHROPIC_API_KEY")

    def prompt_for(self, text, status):
        keys = ("style", "section", "section_bars_left", "energy", "energy_bias", "density", "swing",
                "brightness", "bpm", "key", "camelot", "layers", "muted", "chords", "movement", "state",
                "pattern_slots")
        snap = {k: status.get(k) for k in keys if k in status}
        recent = "\n".join(f"- operator: {t!r} -> {say!r}" for t, say, _ in self.history[-4:])
        return (f"Current state (JSON): {json.dumps(snap, default=str)}\n"
                f"Recent exchanges:\n{recent or '- none'}\n\n"
                f"Operator says: {text.strip()}\n\nReply with the Intent JSON only.")

    def intent_for(self, text, status, slots=SLOTS, sandbox=None):
        reply = self.call(SYSTEM_PROMPT, self.prompt_for(text, status))
        raw = _extract_json(reply)
        intent, warn = validate_intent(raw, slots=slots, sandbox=sandbox)
        return intent, warn, reply
