"""Real instruments and effects: VST3 plugins hosted through pedalboard.

Manifest (first hit wins):
  1. projects/<id>/media/gen/plugins/plugins.json
  2. media/plugins/plugins.json
    {"dexed": {"path": "Dexed.vst3", "kind": "instrument", "program": 0,
               "params": {"cutoff": 0.8}, "preset": "epiano.vstpreset", "tags": ["fm"]}, ...}
  paths relative to the manifest folder or absolute. tools/gen/plugins.py
  scans a folder into a manifest and test-renders a plugin.

A style slot uses one as  {"voice": "vst", "plugin": "vst:dexed", "program": 3,
"params": {...}, "gain": 0.4, ...}. Instruments are STATEFUL, so the
rack renders a VST slot per phrase (all its notes in one call, plus a
release tail) on the conductor thread; the audio thread only mixes the
result. Effects (`VstFx`) run per block on the audio thread with state
carried across calls.

pedalboard is optional: without it (or without the plugin file) a slot
falls back to its "fallback" patch, or to the analog voice named in the
style, so a show never depends on a binary being present."""
from __future__ import annotations

import glob
import json
import os
import sys
import threading

import numpy as np

from lib.gen import RATE

_cache = None
_load_lock = threading.Lock()
_instances = {}


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))          # lib/gen/synth
    return os.path.dirname(os.path.dirname(os.path.dirname(here)))


def _active_project():
    try:
        import yaml
        with open(os.path.join(_repo_root(), "active_project.yaml"), encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        return d.get("active") or d.get("project") or d.get("id")
    except Exception:
        return None


def manifests():
    global _cache
    if _cache is not None:
        return _cache
    out = []
    root = _repo_root()
    cands = []
    pid = _active_project()
    if pid:
        cands.append(os.path.join(root, "projects", str(pid), "media", "gen", "plugins"))
    cands.append(os.path.join(root, "media", "plugins"))
    for folder in cands:
        path = os.path.join(folder, "plugins.json")
        try:
            with open(path, encoding="utf-8") as fh:
                out.append((folder, json.load(fh)))
        except Exception:
            continue
    _cache = out
    return out


def reload():
    global _cache
    _cache = None
    _instances.clear()


def available() -> bool:
    try:
        import pedalboard  # noqa: F401
        return True
    except Exception:
        return False


def binary_path(path: str):
    """A .vst3 bundle folder -> the loadable binary inside it (pedalboard
    scans the inner file, not the bundle, on Windows); a file -> itself."""
    if os.path.isdir(path):
        arch = {"win32": "x86_64-win", "darwin": "MacOS"}.get(sys.platform, "x86_64-linux")
        cands = glob.glob(os.path.join(path, "Contents", arch, "*.vst3")) + \
            glob.glob(os.path.join(path, "Contents", "*", "*.vst3")) + glob.glob(os.path.join(path, "Contents", "*", "*.so"))
        return cands[0] if cands else path
    return path


def resolve(ref: str):
    """"vst:<name>" -> (entry dict with absolute "path", folder) or (None, None)."""
    if not ref or not ref.startswith("vst:"):
        return None, None
    name = ref.split(":", 1)[1].strip()
    for folder, man in manifests():
        entry = man.get(name)
        if entry:
            e = dict(entry)
            p = e.get("path", name + ".vst3")
            e["path"] = p if os.path.isabs(p) else os.path.join(folder, p)
            e["name"] = name
            return e, folder
    return None, None


def names(kind: str | None = None):
    out = []
    for _, man in manifests():
        for k, v in man.items():
            if (kind is None or v.get("kind", "instrument") == kind) and k not in out:
                out.append(k)
    return out


class VstInstrument:
    """One hosted instrument. render(events, t0, seconds) -> (n, 2) float32."""

    def __init__(self, entry: dict, program=None, params=None, preset=None):
        from pedalboard import VST3Plugin
        self.entry = entry
        self.plugin = VST3Plugin(binary_path(entry["path"]))
        self.name = entry.get("name", "?")
        self.lock = threading.Lock()
        self._dirty = False
        self.flush_s = float(entry.get("flush_s", 3.0))
        pre = preset or entry.get("preset")
        if pre:
            p = pre if os.path.isabs(pre) else os.path.join(os.path.dirname(entry["path"]), pre)
            try:
                self.plugin.load_preset(p)
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] vst {self.name}: preset {pre!r} failed ({e})")
        prog = program if program is not None else entry.get("program")
        if prog is not None:
            try:
                self.plugin.program = int(prog) if not isinstance(prog, str) else prog
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] vst {self.name}: program {prog!r} failed ({e})")
        for k, v in dict(entry.get("params") or {}, **(params or {})).items():
            try:
                setattr(self.plugin, k, v)
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] vst {self.name}: param {k}={v!r} failed ({e})")
        try:
            self.latency = int(self.plugin.reported_latency_samples or 0)
        except Exception:
            self.latency = 0

    def render(self, events, t0: int, seconds: float, vel_curve=1.0):
        """events: NoteEvents of ONE slot (at/dur in samples, absolute);
        rendered from absolute sample t0 for `seconds`."""
        from mido import Message
        msgs = []
        for e in events:
            on = max(0.0, (e.at - t0) / RATE)
            off = max(on + 0.01, (e.at + e.dur - t0) / RATE)
            note = int(round(e.pitch))
            vel = int(max(1, min(127, round(127 * (0.25 + 0.75 * min(1.0, e.vel) ** vel_curve)))))
            msgs.append(Message("note_on", note=note, velocity=vel, time=on))
            msgs.append(Message("note_off", note=note, time=off))
        msgs.sort(key=lambda m: m.time)
        with self.lock:
            # reset=False: pedalboard reloads the plugin on reset and refuses to do that off the
            # main thread (the conductor and the console's workers render here). State carries
            # between calls instead, so flush the previous call's tail with silence first - the
            # batch must start clean at its own t0, not with a ghost of the last phrase.
            if self._dirty:
                self.plugin([], duration=float(self.flush_s), sample_rate=float(RATE), num_channels=2, reset=False)
            audio = self.plugin(msgs, duration=float(seconds), sample_rate=float(RATE), num_channels=2, reset=False)
            self._dirty = bool(msgs)
        return np.ascontiguousarray(audio.T, dtype=np.float32)


class VstFx:
    """One hosted effect on a bus, state carried across blocks."""

    def __init__(self, entry: dict, params=None, preset=None):
        from pedalboard import VST3Plugin
        self.entry = entry
        self.plugin = VST3Plugin(binary_path(entry["path"]))
        self.name = entry.get("name", "?")
        pre = preset or entry.get("preset")
        if pre:
            p = pre if os.path.isabs(pre) else os.path.join(os.path.dirname(entry["path"]), pre)
            try:
                self.plugin.load_preset(p)
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] vst fx {self.name}: preset failed ({e})")
        for k, v in dict(entry.get("params") or {}, **(params or {})).items():
            try:
                setattr(self.plugin, k, v)
            except Exception:  # noqa: BLE001
                pass
        self.mix = 1.0

    def process(self, x):
        y = self.plugin(np.ascontiguousarray(x.T, dtype=np.float32), float(RATE), reset=False)
        y = y.T
        if y.shape[0] != x.shape[0]:
            out = np.zeros_like(x)
            n = min(y.shape[0], x.shape[0])
            out[:n] = y[:n]
            y = out
        return y.astype(np.float32) if self.mix >= 1.0 else (x * (1.0 - self.mix) + y * self.mix).astype(np.float32)


def instrument(patch: dict):
    """The VstInstrument a slot patch asks for, or None (missing plugin /
    no pedalboard). Instances are cached per (plugin, program, preset,
    params) so a style swap back and forth does not reload binaries."""
    if not available():
        return None
    entry, _ = resolve(str(patch.get("plugin", "")))
    if entry is None or not os.path.exists(entry["path"]):
        return None
    key = (entry["path"], str(patch.get("program", entry.get("program"))), str(patch.get("preset", entry.get("preset"))),
           json.dumps(patch.get("params") or {}, sort_keys=True))
    with _load_lock:
        inst = _instances.get(key)
        if inst is None:
            try:
                inst = VstInstrument(entry, program=patch.get("program"), params=patch.get("params"), preset=patch.get("preset"))
            except Exception as e:  # noqa: BLE001
                print(f"[GEN] vst {entry.get('name')}: load failed ({type(e).__name__}: {e})")
                return None
            _instances[key] = inst
    return inst


def effect(spec: dict):
    if not available():
        return None
    entry, _ = resolve(str(spec.get("plugin", "")))
    if entry is None or not os.path.exists(entry["path"]):
        return None
    try:
        fx = VstFx(entry, params=spec.get("params"), preset=spec.get("preset"))
        fx.mix = float(spec.get("mix", 1.0))
        return fx
    except Exception as e:  # noqa: BLE001
        print(f"[GEN] vst fx {entry.get('name')}: load failed ({type(e).__name__}: {e})")
        return None


def overlay(style: dict) -> dict:
    """Apply the style's optional "vst" overlay ({slot: patch}) to a COPY
    of the style: slots whose plugin resolves become "vst" voices (the
    original patch is kept as the fallback), others are left alone."""
    ov = style.get("vst") or {}
    if not ov:
        return style
    import copy
    st = copy.deepcopy(style)
    for slot, vp in ov.items():
        base = st["slots"].get(slot)
        if base is None:
            continue
        if instrument(vp) is None:
            continue
        merged = dict(base)
        merged.update(vp)
        merged["voice"] = "vst"
        merged["fallback"] = {k: v for k, v in base.items() if k != "layers"}
        st["slots"][slot] = merged
    return st
