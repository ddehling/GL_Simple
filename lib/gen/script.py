"""The song description language: a SongScript is the list of commands
that makes the generator play a particular song - style, tempo, key, and
a sequence of sections each with its length and the levers the operator
(or the analyser) sets for it.

    title: Night Signal
    style: groove          # any STYLES name
    bpm: 124
    key: 8A                # Camelot or "A minor"
    seed: 7
    humanize: 1.0
    end: true              # after the last section: outro and stop (else the grammar continues)
    sections:
      - section: intro     # generator section name (intro/groove/build/drop/break/outro; flow/swell/calm for ambient)
        bars: 16           # multiple of 4
        energy: 0.35       # 0..1 target (overrides the form)
        density: 1.0       # 0..1.5
        brightness: 1.0    # 0.4..1.6
        swing: 0.08        # 0..0.33
        layers: [kick, hat, pad]        # exactly these slots play
        chords: [0, 0, 5, 6]            # scale degrees per bar, looped (plain: no borrowed/sus colour)
        lanes: {lp: 20000, verb: 1.2}   # mix lanes at the section start
        key: 9A            # change at this section
        bpm: 126
        hook: {steps: [0, 3, 6, 10, 12], degrees: [0, 2, 4, 2, 0], contour: arch}   # the theme from here on

Scripts are YAML (or JSON). The composer follows one through
Composer.load_script (form, harmony, melody, mix lanes); render() plays
one offline to audio - the analyser scores its recreation of an ingested
song with it (lib/gen/analysis)."""
from __future__ import annotations

import copy
import json
import os

import numpy as np

from lib.gen import RATE

SECTION_KEYS = ("section", "bars", "energy", "density", "brightness", "swing", "layers", "chords", "lanes", "key", "bpm", "hook", "bass",
                "drums", "drums_grid")
# drums: {"kick": [[step, vel], ...], "snare": [...], "hat": [...]}  the section's beat, as hits on the 16th grid (the kit plays exactly this)
# drums_grid: {"kick": [16 floats], ...}  the folded onset strengths it was read from (evidence, shown in the tab)
# bass: {"steps": [16th onsets], "degrees": [offsets from the tonic]}  the section's bass cell (from the source's bass stem)
DEFAULT = {"title": "", "style": "groove", "bpm": None, "key": "8A", "seed": 1, "humanize": 1.0, "end": True, "sections": [],
           "kit": None, "vocals": [], "bank": [], "bpm_src": None}
# bank:    [{"file": wav, "base_midi": 64}]  pitched slices of the source's melodic stem (keys/arp play them)
# bpm_src: the source song's tempo (vocal phrases are time-stretched by bpm/bpm_src)
# kit:    {"kick": wav, "snare": wav, "hat": wav}  one-shots cut from the source song (the recreation plays them)
# vocals: [{"bar": 12.25, "file": wav, "seconds": 1.8}]  the source song's vocal phrases, placed on its bar grid


def normalize(script: dict) -> dict:
    """A copy with defaults filled and section lengths on the 4-bar grid."""
    s = dict(DEFAULT)
    s.update({k: v for k, v in (script or {}).items() if k in DEFAULT})
    out = []
    for e in (script or {}).get("sections") or []:
        e = {k: v for k, v in dict(e).items() if k in SECTION_KEYS}
        e["section"] = str(e.get("section", "groove"))
        e["bars"] = max(4, int(round(float(e.get("bars", 8)) / 4.0)) * 4)
        for k in ("energy", "density", "brightness", "swing"):
            if k in e and e[k] is not None:
                e[k] = float(e[k])
        if "layers" in e and e["layers"] is not None:
            e["layers"] = [str(x) for x in e["layers"]]
        if "chords" in e and e["chords"] is not None:
            e["chords"] = [int(x) for x in e["chords"]] or None
        out.append(e)
    s["sections"] = out
    s["kit"] = {k: str(v) for k, v in (s.get("kit") or {}).items() if v} or None
    s["vocals"] = [{"bar": float(v["bar"]), "file": str(v["file"]), "seconds": float(v.get("seconds", 1.0))}
                   for v in (s.get("vocals") or []) if v.get("file")]
    s["bank"] = [{"file": str(b["file"]), "base_midi": int(b.get("base_midi", 60))} for b in (s.get("bank") or []) if b.get("file")]
    s["bpm_src"] = float(s["bpm_src"]) if s.get("bpm_src") else None
    s["seed"] = int(s.get("seed") or 1)
    s["bpm"] = float(s["bpm"]) if s.get("bpm") else None
    return s


def total_bars(script: dict) -> int:
    return sum(int(e.get("bars", 8)) for e in (script.get("sections") or []))


def total_seconds(script: dict, bpm: float | None = None) -> float:
    bpm = bpm or script.get("bpm") or 120.0
    return total_bars(script) * 4 * 60.0 / float(bpm)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if path.lower().endswith(".json"):
        return normalize(json.loads(text))
    import yaml
    return normalize(yaml.safe_load(text) or {})


def save(script: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    s = normalize(script)
    with open(path, "w", encoding="utf-8") as fh:
        if path.lower().endswith(".json"):
            json.dump(s, fh, indent=1)
        else:
            import yaml
            yaml.safe_dump(s, fh, sort_keys=False, allow_unicode=True)
    return path


KIT_BASE = {"kick": 36, "snare": 38, "hat": 42, "ohat": 46, "perc": 60, "shaker": 70, "ride": 51, "tom": 48, "rim": 37}


def apply_material(style: dict, script: dict) -> dict:
    """A copy of the style whose kit slots play the script's one-shots and
    which has a "vox" slot when the script places vocal phrases."""
    s = normalize(script)
    if not s.get("kit") and not s.get("vocals") and not s.get("bank"):
        return style
    st = copy.deepcopy(style)
    for slot, path in (s.get("kit") or {}).items():
        if slot in st["slots"] and os.path.exists(path):
            base = st["slots"][slot]
            st["slots"][slot] = {"voice": "sample", "file": path, "base_midi": KIT_BASE.get(slot, 60), "gain": float(base.get("gain", 0.8)),
                                 "send_reverb": float(base.get("send_reverb", 0.0)), "octave": base.get("octave", 3)}
    bank = [b for b in (s.get("bank") or []) if os.path.exists(b["file"])]
    if bank:
        for slot in ("keys", "arp"):
            if slot in st["slots"]:
                base = st["slots"][slot]
                st["slots"][slot] = {"voice": "sample", "samples": bank, "gain": float(base.get("gain", 0.4)) * 0.9,
                                     "send_reverb": float(base.get("send_reverb", 0.3)), "send_delay": float(base.get("send_delay", 0.0)),
                                     "octave": base.get("octave", 3), "decay": 0.6}
    if s.get("vocals"):
        st["slots"]["vox"] = {"voice": "sample", "gain": 0.85, "base_midi": 60, "send_reverb": 0.2}
        for sec in st["sections"].values():
            if "*" not in sec["layers"]:
                sec["layers"] = set(sec["layers"]) | {"vox"}
    return st


def make_composer(script: dict, arc_fn=None):
    from lib.gen.composer import Composer
    s = normalize(script)
    c = Composer(s["style"], bpm=s["bpm"], key=s["key"], seed=s["seed"], arc_fn=arc_fn)
    c.load_script(s)
    return c


def render(script: dict, out_path: str | None = None, seconds: float | None = None, seed: int | None = None,
           progress=None):
    """Play the script offline. Returns (audio (n,2) float32, composer).
    seconds defaults to the script's own length (plus a bar of tail)."""
    from lib.gen.synth import SynthRack
    s = normalize(script)
    if seed is not None:
        s["seed"] = int(seed)
    c = make_composer(s)
    total = float(seconds) if seconds else total_seconds(s, c.bpm) + 4 * 60.0 / c.bpm
    rack = SynthRack(apply_material(c.style, s), c.bpm, seed=s["seed"])
    rack.warm_up()
    n_total = int(total * RATE)
    for p in c.phrases_until(n_total):
        rack.schedule(p.events)
    out = []
    done = 0
    while rack.clock < n_total:
        out.append(rack.render(2048))
        done += 2048
        if progress is not None and done % (2048 * 64) == 0:
            progress(min(1.0, done / n_total))
    audio = np.concatenate(out)[:n_total]
    if out_path:
        import soundfile as sf
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        sf.write(out_path, np.clip(audio, -1, 1), RATE, subtype="PCM_16")
    return audio, c


def to_actions(script: dict, slots=None) -> list:
    """The script as the system's own command list: [(bar, action, value)]
    using the whitelisted gen actions (lib/gen/actions.py), so an operator
    (or the analyser) can read a song as the steering that produces it."""
    from lib.gen.events import SLOTS
    s = normalize(script)
    out = [(0, "style", s["style"]), (0, "key", s["key"])]
    if s.get("bpm"):
        out.append((0, "bpm", float(s["bpm"])))
    out.append((0, "humanize", float(s.get("humanize", 1.0))))
    bar = 0
    all_slots = list(slots or SLOTS)
    prev_layers = None
    for e in s["sections"]:
        out.append((bar, "section", e["section"]))
        if e.get("bpm"):
            out.append((bar, "bpm", float(e["bpm"])))
        if e.get("key"):
            out.append((bar, "key", str(e["key"])))
        if e.get("energy") is not None:
            out.append((bar, "energy", round(float(e["energy"]) - 0.6, 3)))   # bias around the form's middle
        for k in ("density", "swing", "brightness"):
            if e.get(k) is not None:
                out.append((bar, k, float(e[k])))
        if e.get("layers") is not None and e["layers"] != prev_layers:
            want = set(e["layers"])
            for sl in all_slots:
                if sl in ("auto", "fx"):
                    continue
                out.append((bar, "mute", {"slot": sl, "on": sl not in want}))
            prev_layers = e["layers"]
        for lane, v in (e.get("lanes") or {}).items():
            out.append((bar, "lane", {"lane": lane, "to": float(v), "ramp_s": 2.0}))
        if e.get("chords"):
            out.append((bar, "chords", list(e["chords"])))          # (script-only: the action list has no chord lever)
        if e.get("hook"):
            out.append((bar, "hook", e["hook"].get("name", "hook")))   # (script-only)
        if e.get("drums"):
            out.append((bar, "drums", {k: [st for st, _ in v] for k, v in e["drums"].items()}))   # (script-only: the beat)
        bar += e["bars"]
    if s.get("end", True):
        out.append((bar, "end", None))
    return out


def describe(script: dict) -> str:
    """One line per section, for logs and the console."""
    s = normalize(script)
    lines = [f"{s.get('title') or 'untitled'}: {s['style']} {s['bpm'] or '?'} bpm {s['key']} seed {s['seed']}"]
    bar = 0
    for e in s["sections"]:
        bits = [f"bar {bar:4d}", f"{e['section']:7s}", f"{e['bars']:3d} bars"]
        for k in ("energy", "density", "brightness", "swing"):
            if k in e:
                bits.append(f"{k[:3]} {e[k]:.2f}")
        if e.get("layers"):
            bits.append("+".join(e["layers"]))
        if e.get("chords"):
            bits.append("chords " + " ".join(str(x) for x in e["chords"]))
        if e.get("key") or e.get("bpm"):
            bits.append(f"-> {e.get('key', '')} {e.get('bpm', '')}".strip())
        if e.get("hook"):
            bits.append("hook")
        if e.get("drums"):
            bits.append("beat " + "/".join(f"{k}:{''.join('x' if any(st == i for st, _ in v) else '.' for i in range(16))}" for k, v in e["drums"].items() if v))
        lines.append("  " + "  ".join(bits))
        bar += e["bars"]
    if s.get("kit"):
        lines.append("  kit: " + ", ".join(f"{k}={os.path.basename(v)}" for k, v in s["kit"].items()))
    if s.get("vocals"):
        lines.append(f"  vocals: {len(s['vocals'])} phrases placed (first at bar {s['vocals'][0]['bar']})")
    if s.get("bank"):
        lines.append(f"  bank: {len(s['bank'])} melodic tones (keys/arp play the song's own sound)")
    if any(e.get("bass") for e in s["sections"]):
        lines.append("  bass: cells transcribed from the bass stem")
    return "\n".join(lines)


def example() -> dict:
    return normalize({
        "title": "example", "style": "groove", "bpm": 124, "key": "8A", "seed": 3,
        "sections": [
            {"section": "intro", "bars": 8, "energy": 0.3, "layers": ["kick", "hat", "pad"]},
            {"section": "groove", "bars": 16, "energy": 0.6, "chords": [0, 0, 5, 6]},
            {"section": "build", "bars": 8, "energy": 0.8},
            {"section": "drop", "bars": 16, "energy": 1.0, "hook": {"steps": [0, 3, 6, 10, 12, 16, 22, 28],
                                                                      "degrees": [0, 2, 4, 2, 0, 4, 2, 0], "contour": "arch"}},
            {"section": "break", "bars": 8, "energy": 0.35, "lanes": {"lp": 3000, "verb": 1.8}},
            {"section": "outro", "bars": 8, "energy": 0.25},
        ],
    })


def _deep(d):
    return copy.deepcopy(d)
