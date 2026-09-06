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

Two engines, one interface (eval / query / ping / stop):
  * StrudelV8   IN-PROCESS: tools/gen/strudel/dist/strudel.bundle.js (the four
                packages bundled by esbuild) inside an embedded V8 via the
                `mini-racer` wheel (or the older `py-mini-racer`). No node on
                the show box. Default when the wheel is importable.
  * StrudelBridge  a Node subprocess running bridge.mjs (needs node >= 18 and
                `npm install` in tools/gen/strudel). Fallback / development.
open_engine() picks the first that works.
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


BUNDLE = os.path.join(BRIDGE_DIR, "dist", "strudel.bundle.js")


def _racer():
    """The embedded-V8 class, from whichever wheel is installed."""
    try:
        from mini_racer import MiniRacer
        return MiniRacer
    except Exception:
        try:
            from py_mini_racer import MiniRacer
            return MiniRacer
        except Exception:
            return None


def v8_available():
    if _racer() is None:
        return False, "pip install mini-racer"
    if not os.path.exists(BUNDLE):
        return False, f"missing {BUNDLE} (tools/gen/strudel/build_bundle.sh)"
    return True, ""


def node_available():
    if not shutil.which("node"):
        return False, "node not found on PATH"
    if not os.path.isdir(os.path.join(BRIDGE_DIR, "node_modules", "@strudel", "core")):
        return False, f"run `npm install` in {BRIDGE_DIR}"
    return True, ""


def available():
    """(ok, reason) - some engine can run."""
    ok, why = v8_available()
    if ok:
        return True, ""
    ok2, why2 = node_available()
    if ok2:
        return True, ""
    return False, f"{why}; or {why2}"


def open_engine(prefer="v8"):
    """Start and return the first working engine (V8 in-process first)."""
    order = ("v8", "node") if prefer == "v8" else ("node", "v8")
    errors = []
    for name in order:
        try:
            eng = StrudelV8() if name == "v8" else StrudelBridge()
            if eng.start():
                return eng
            errors.append(f"{name}: ping failed")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")
    raise RuntimeError("no Strudel engine: " + "; ".join(errors))


_V8_PRELUDE = """
var console = {log:function(){}, warn:function(){}, error:function(){}, info:function(){}, debug:function(){}};
if (typeof performance === 'undefined') var performance = { now: function(){ return Date.now(); } };
var __timers = [];
if (typeof setTimeout === 'undefined') {
  var setTimeout = function(fn, ms){ __timers.push(fn); return __timers.length; };
  var clearTimeout = function(id){ __timers[id-1] = null; };
  var setInterval = function(){ return 0; }; var clearInterval = function(){};
}
function __runTimers(){ var t = __timers; __timers = []; for (var i=0;i<t.length;i++) if (t[i]) t[i](); }
"""
_V8_SETUP = """
var __ready = false, __err = null, __pattern = null, __evalDone = null;
(async () => { try {
  const {core, mini, tonal} = __strudel;
  await core.evalScope(core, mini, tonal);
  const ctx = { energy: 0.5, section: 'groove', bar: 0, key: '8A', bpm: 120, phrase: 0, chords: [] };
  globalThis.ctx = ctx;
  for (const k of ['energy', 'bar', 'bpm', 'phrase']) globalThis[k] = core.signal(() => Number(ctx[k]));
  globalThis.section = () => ctx.section;
  globalThis.key = () => ctx.key;
  globalThis.__eval = (code) => { __evalDone = null;
    core.evaluate(code, __strudel.transpiler)
      .then(r => { if (!r || !r.pattern || typeof r.pattern.queryArc !== 'function') throw new Error('code did not produce a pattern');
                   r.pattern.queryArc(0, 1); __pattern = r.pattern; __evalDone = true; })
      .catch(e => { __evalDone = 'error: ' + String(e && e.message ? e.message : e); }); };
  globalThis.__query = (a, b, c) => { Object.assign(ctx, c || {}); if (!__pattern) return '[]';
    return JSON.stringify(__pattern.queryArc(a, b).filter(h => h.hasOnset && h.hasOnset())
      .map(h => ({ b: Number(h.whole.begin.valueOf()), e: Number(h.whole.end.valueOf()), v: h.value }))); };
  __ready = true;
} catch (e) { __err = String(e && e.stack || e); } })();
"""


class StrudelV8:
    """Strudel inside the Python process (embedded V8). Same interface as
    StrudelBridge. Async JS work is driven by polling a done-flag while
    flushing shimmed timers - no event loop needed."""

    def __init__(self, timeout=5.0):
        self.timeout = float(timeout)
        self._ctx = None
        self._lock = threading.Lock()

    def start(self):
        ok, why = v8_available()
        if not ok:
            raise RuntimeError(f"Strudel V8 unavailable: {why}")
        Racer = _racer()
        ctx = Racer()
        ctx.eval(_V8_PRELUDE)
        with open(BUNDLE, encoding="utf-8") as fh:
            ctx.eval(fh.read())
        ctx.eval(_V8_SETUP)
        self._ctx = ctx
        self._wait("__ready", "__err")
        if ctx.eval("__err"):
            raise RuntimeError(f"Strudel V8 setup failed: {ctx.eval('__err')}")
        return True

    def _wait(self, flag, err_flag=None):
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            self._ctx.eval("__runTimers()")
            v = self._ctx.eval(flag)
            if v is not None and v is not False:
                return v
            if err_flag and self._ctx.eval(err_flag):
                return None
            time.sleep(0.005)
        raise TimeoutError("Strudel V8 timed out")

    @property
    def alive(self):
        return self._ctx is not None

    def eval(self, code):
        with self._lock:
            self._ctx.eval("__eval(%s)" % json.dumps(code))
            d = self._wait("__evalDone")
        if isinstance(d, str) and d.startswith("error:"):
            raise ValueError(d[6:].strip())
        return True

    def query(self, cycle_from, cycle_to, ctx=None):
        with self._lock:
            raw = self._ctx.eval("__query(%r, %r, %s)" % (float(cycle_from), float(cycle_to), json.dumps(ctx or {})))
        return json.loads(raw)

    def request(self, obj):
        if obj.get("op") == "ping":
            return {"ok": True}
        raise NotImplementedError

    def stop(self):
        self._ctx = None


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
        ok, why = node_available()
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
