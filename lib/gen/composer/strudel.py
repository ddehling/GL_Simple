"""Strudel as a composer: pattern code -> events on the rack's sample clock.

Strudel (https://strudel.cc, the JavaScript port of TidalCycles) is a
pattern language: `s("bd*4, [~ cp]*2, hh(5,8)")` is a bar of drums,
`note("<0 3 5 7>(3,8)").scale("A:minor")` a Euclidean bass line. Its
engine is pure JavaScript, so it runs headless under Node: tools/gen/
strudel/bridge.mjs evaluates code and answers "which events fall in
cycles [a, b)" over JSON lines. Here, ONE CYCLE == ONE BAR at the
composer's tempo, and every event becomes a NoteEvent for the same
SynthRack the rule composer drives - SoundFonts, SuperCollider, the show
integration and the visuals are unchanged.

Mapping (Strudel control -> rack):
  s        sample/sound name -> slot: bd/kick->kick, sd/sn/cp/clap->snare,
           hh/hat/ch->hat, oh/ohat->ohat, rim/perc/lt/mt/ht/tom->perc,
           bass/lead/pad/arp/keys -> that slot; unknown + note -> lead
  note/n   MIDI number or name ("c3", "eb4") -> pitch (drums ignore it)
  gain/velocity -> velocity;  cutoff/lpf -> params.cutoff;  legato/clip ->
           gate length scale;  vowel/room/pan/delay are ignored for now.

Needs: node >= 18 and `npm install` in tools/gen/strudel (once).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time

from lib.gen import RATE
from lib.gen.events import NoteEvent, SLOTS
from lib.gen.theory import _PC

BRIDGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "tools", "gen", "strudel")

_SLOT_ALIASES = {
    "bd": "kick", "kick": "kick", "bassdrum": "kick",
    "sd": "snare", "sn": "snare", "cp": "snare", "clap": "snare", "snare": "snare",
    "hh": "hat", "hat": "hat", "ch": "hat", "hihat": "hat",
    "oh": "ohat", "ohat": "ohat", "openhat": "ohat",
    "rim": "perc", "perc": "perc", "lt": "perc", "mt": "perc", "ht": "perc", "tom": "perc",
    "rs": "perc", "cb": "perc", "sh": "perc", "tb": "perc",
    "bass": "bass", "lead": "lead", "pad": "pad", "arp": "arp", "keys": "keys",
    "piano": "keys", "superpiano": "keys", "supersaw": "lead", "sawtooth": "lead",
    "square": "lead", "triangle": "lead", "sine": "pad",
}
_NOTE_RE = re.compile(r"^([a-gA-G])([#b]*)(-?\d+)?$")


def available():
    """(ok, reason) - node present and the bridge's npm install done."""
    if not shutil.which("node"):
        return False, "node not found on PATH"
    if not os.path.isdir(os.path.join(BRIDGE_DIR, "node_modules", "@strudel", "core")):
        return False, f"run `npm install` in {BRIDGE_DIR}"
    return True, ""


def note_to_midi(v, default_octave=3):
    if isinstance(v, (int, float)):
        return float(v)
    m = _NOTE_RE.match(str(v).strip())
    if not m:
        return None
    letter, acc, octv = m.groups()
    pc = _PC[letter.upper()]
    pc += acc.count("#") - acc.count("b")
    octv = int(octv) if octv is not None else default_octave
    return float(12 * (octv + 1) + pc)


class StrudelBridge:
    """One Node process; requests matched by id; restarted if it dies."""

    def __init__(self, timeout=5.0):
        self.timeout = float(timeout)
        self._p = None
        self._lock = threading.Lock()
        self._id = 0
        self._pending = {}
        self._reader = None
        self.last_stderr = ""

    def start(self):
        ok, why = available()
        if not ok:
            raise RuntimeError(f"Strudel bridge unavailable: {why}")
        self._p = subprocess.Popen(["node", "bridge.mjs"], cwd=BRIDGE_DIR, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read, daemon=True, name="strudel-bridge")
        self._reader.start()
        threading.Thread(target=self._drain_err, daemon=True).start()
        return self.request({"op": "ping"}).get("ok", False)

    def _drain_err(self):
        try:
            for line in self._p.stderr:
                self.last_stderr = line.strip()[-200:]
        except Exception:
            pass

    def _read(self):
        try:
            for line in self._p.stdout:
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                ev = self._pending.get(o.get("id"))
                if ev is not None:
                    ev[1] = o
                    ev[0].set()
        except Exception:
            pass

    @property
    def alive(self):
        return self._p is not None and self._p.poll() is None

    def request(self, obj):
        if not self.alive:
            self.start()
        with self._lock:
            self._id += 1
            rid = self._id
        ev = [threading.Event(), None]
        self._pending[rid] = ev
        obj = dict(obj, id=rid)
        try:
            self._p.stdin.write(json.dumps(obj) + "\n")
            self._p.stdin.flush()
        except Exception as e:
            self._pending.pop(rid, None)
            raise RuntimeError(f"bridge write failed: {e}")
        if not ev[0].wait(self.timeout):
            self._pending.pop(rid, None)
            raise TimeoutError("Strudel bridge timed out")
        self._pending.pop(rid, None)
        return ev[1] or {}

    def eval(self, code):
        r = self.request({"op": "eval", "code": code})
        if r.get("error"):
            raise ValueError(r["error"])
        return True

    def query(self, cycle_from, cycle_to, ctx=None):
        r = self.request({"op": "query", "from": cycle_from, "to": cycle_to, "ctx": ctx or {}})
        if r.get("error"):
            raise ValueError(r["error"])
        return r.get("haps", [])

    def stop(self):
        try:
            if self._p:
                self._p.stdin.close()
                self._p.terminate()
        except Exception:
            pass
        self._p = None


class StrudelSource:
    """The Composer's `pattern_source`: converts haps to NoteEvents."""

    def __init__(self, bridge: StrudelBridge, slots):
        self.bridge = bridge
        self.slots = set(slots)
        self.code = None
        self.error = ""

    def load(self, code: str):
        self.bridge.eval(code)       # raises ValueError on bad code
        self.code = code
        self.error = ""

    def events(self, bar0, nbars, start, samples_per_bar, ctx):
        haps = self.bridge.query(bar0, bar0 + nbars, ctx)
        out = []
        for h in haps:
            v = h.get("v") or {}
            if not isinstance(v, dict):
                v = {"s": str(v)}
            slot = self._slot(v)
            if slot is None or slot not in self.slots:
                continue
            b, e = float(h["b"]), float(h["e"])
            at = int(round(start + (b - bar0) * samples_per_bar))
            length = max(1, int((e - b) * samples_per_bar))
            gate = float(v.get("legato", v.get("clip", 1.0)) or 1.0)
            dur = max(int(length * min(4.0, max(0.05, gate))) - 1, int(0.02 * RATE))
            vel = v.get("velocity", v.get("gain", 0.8))
            try:
                vel = float(vel)
            except (TypeError, ValueError):
                vel = 0.8
            pitch = 36.0
            if slot not in ("kick", "snare", "hat", "ohat", "perc"):
                p = note_to_midi(v.get("note", v.get("n", None)) if v.get("note") is not None or v.get("n") is not None else None)
                if p is None:
                    continue
                pitch = p
            params = {}
            cut = v.get("cutoff", v.get("lpf"))
            if cut is not None:
                try:
                    params["cutoff"] = float(cut)
                except (TypeError, ValueError):
                    pass
            out.append(NoteEvent(at, slot, pitch, max(0.0, min(1.0, vel)), dur, params))
        out.sort(key=lambda e: (e.at, e.slot))
        return out

    @staticmethod
    def _slot(v):
        s = v.get("s")
        if isinstance(s, str):
            base = s.split(":")[0].lower()
            if base in _SLOT_ALIASES:
                return _SLOT_ALIASES[base]
            if base in SLOTS:
                return base
        if v.get("note") is not None or v.get("n") is not None:
            return "lead"
        return None
