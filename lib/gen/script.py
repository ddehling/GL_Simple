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
import re

import numpy as np

from lib.gen import RATE

SECTION_KEYS = ("section", "bars", "energy", "density", "brightness", "swing", "layers", "chords", "lanes", "key", "bpm", "hook", "bass",
                "drums", "drums_grid", "drums_phrases", "drums_bars", "dyn", "level", "trim_db", "loops", "melody", "bass_line")
# drums_bars: [{slot: [[step, vel], ...]} per bar of the section]  the identified beat bar by bar (from the drum stem's sounds);
#        the kit plays it as a template per bar (strong hits always, weak ones by strength; thinned only when the steering
#        goes below the section's own energy / density), so the song's fills and variation are there and still steerable
# level: the section's mean level in dB relative to the song's mean (from the source); render(calibrate=True) measures the
#        recreation's own section levels and writes the difference into trim_db (dB, added to the gain lane) - a second pass
# chords: one entry per bar (cycled when shorter than the section): a diatonic degree (int) or {"deg", "third": "maj"|"min", "sus": 0|2|4}
#         when the song's chord quality is not the key's own (the harmony spells it with an altered third / a suspension)
# drums_phrases: [{"kick": [[step, vel]], "snare", "hat", "fill": {...}|None}, ...]  one template per 4-bar phrase of the section
#         (the kit varies it); "fill" is the phrase's last bar when it differs from the rest (a fill), played on that bar
# dyn: [dB per bar]  the section's dynamics relative to its own mean level, written to the "gain" lane bar by bar
# melody / bass_line: [[bar_in_section, step, midi, dur_steps, vel], ...]  the transcribed lines (notes as samples)
# loops: {"drums": wav, "bass": wav, "other": wav, "vocals": wav}  a 4-bar loop per stem cut from the source for this section
# drums: {"kick": [[step, vel], ...], "snare": [...], "hat": [...]}  the section's beat, as hits on the 16th grid (the kit plays exactly this)
# drums_grid: {"kick": [16 floats], ...}  the folded onset strengths it was read from (evidence, shown in the tab)
# bass: {"steps": [16th onsets], "degrees": [offsets from the tonic]}  the section's bass cell (from the source's bass stem)
DEFAULT = {"title": "", "style": "groove", "bpm": None, "key": "8A", "seed": 1, "humanize": 1.0, "end": True, "sections": [],
           "kit": None, "vocals": [], "bank": [], "bass_bank": [], "bpm_src": None, "fidelity": 0.0,
           "motifs": [], "bass_cells": [], "pad": None, "bank_keys": [], "mix_db": {}, "kit_db": {},
           "level_ref_db": None, "master_db": 0.0, "fx": True}
# fx: the form's own transition material (risers, impacts, reverse cymbals, sweeps); the analyser writes False -
#     a recreation plays the song's material, and a riser at -6 dB under it is a wall of noise
# level_ref_db: the source's RMS (dBFS); render's calibration pass sets master_db so the recreation's RMS matches
#               it as far as the peaks allow (at most ~2 dB into the limiter) - the record's level without crushing it
# kit_db: {slot: dB}  each identified drum sound's level relative to the loudest one (the kit's balance, from the song)
# mix_db: {"drums", "bass", "other", "vocals": dB}  per-stem trims, measured by score --stems against the source's stems
# pad:       {"file", "base_midi", "seconds"}  the song's sustained texture (its steadiest two bars), the pad slot plays it
# bank_keys: [{"file", "base_midi"}]  a second identified plucked instrument (the keys slot plays it)
# kit may carry any drum slot (kick, snare, hat, ohat, shaker, perc, rim, tom, ride): the song's identified sounds
# motifs:     [{"steps", "degrees", "contour", "count", "name"}]  the song's melody cells -> the generator's motif memory
# bass_cells: [{"steps", "degrees", "count"}]                      the song's bass cells -> the bass generator's library
# bass_bank: [{"file", "base_midi"}]  the bass stem's notes as samples (the bass slot plays them)
# fidelity: how much of the recreation is the source's own loops: 0 = generator only, 0.5 = drum + bass loops under generated
#           melodic layers, 1 = every loop that exists, the generator only fills what the loops lack (transitions, hook)
# bank:    [{"file": wav, "base_midi": 64}]  pitched slices of the source's melodic stem (keys/arp play them)
# bpm_src: the source song's tempo (vocal phrases are time-stretched by bpm/bpm_src)
# kit:    {"kick": wav, "snare": wav, "hat": wav}  one-shots cut from the source song (the recreation plays them)
# vocals: [{"bar": 12.25, "file": wav, "seconds": 1.8}]  the source song's vocal phrases, placed on its bar grid


def chord_deg(c) -> int:
    """The degree of a chord entry (int or dict)."""
    return int(c["deg"]) if isinstance(c, dict) else int(c)


def parse_chord(x):
    """Chord entry from an int, a dict or its text form: "3", "3M" (major
    third), "3m" (minor), "3s4" / "3s2" (suspended), combinable ("5Ms2")."""
    if isinstance(x, dict):
        d = {"deg": int(x.get("deg", 0))}
        if x.get("third") in ("maj", "min"):
            d["third"] = x["third"]
        if int(x.get("sus", 0) or 0) in (2, 4):
            d["sus"] = int(x["sus"])
        return d if len(d) > 1 else d["deg"]
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip()
    m = re.match(r"^(-?\d+)([Mm])?(?:s([24]))?$", s)
    if not m:
        return int(float(s))
    d = {"deg": int(m.group(1))}
    if m.group(2):
        d["third"] = "maj" if m.group(2) == "M" else "min"
    if m.group(3):
        d["sus"] = int(m.group(3))
    return d if len(d) > 1 else d["deg"]


def chord_str(c) -> str:
    if not isinstance(c, dict):
        return str(int(c))
    return f"{int(c['deg'])}{'M' if c.get('third') == 'maj' else ('m' if c.get('third') == 'min' else '')}{'s' + str(c['sus']) if c.get('sus') else ''}"


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
            e["chords"] = [parse_chord(x) for x in e["chords"]] or None
        if e.get("dyn") is not None:
            e["dyn"] = [round(float(max(-12.0, min(12.0, x))), 2) for x in e["dyn"]] or None
        for k in ("level", "trim_db"):
            if e.get(k) is not None:
                e[k] = round(float(max(-18.0, min(18.0, e[k]))), 2)
        if e.get("drums_phrases") is not None:
            e["drums_phrases"] = [dict(p) for p in e["drums_phrases"]] or None
        if e.get("drums_bars") is not None:
            e["drums_bars"] = [dict(p) for p in e["drums_bars"]] or None
        out.append(e)
    s["sections"] = out
    s["kit"] = {k: str(v) for k, v in (s.get("kit") or {}).items() if v} or None
    s["vocals"] = [{"bar": float(v["bar"]), "file": str(v["file"]), "seconds": float(v.get("seconds", 1.0))}
                   for v in (s.get("vocals") or []) if v.get("file")]
    s["bank"] = [{"file": str(b["file"]), "base_midi": int(b.get("base_midi", 60))} for b in (s.get("bank") or []) if b.get("file")]
    s["bass_bank"] = [{"file": str(b["file"]), "base_midi": int(b.get("base_midi", 36))} for b in (s.get("bass_bank") or []) if b.get("file")]
    s["bank_keys"] = [{"file": str(b["file"]), "base_midi": int(b.get("base_midi", 60))} for b in (s.get("bank_keys") or []) if b.get("file")]
    s["mix_db"] = {str(k): round(float(max(-18.0, min(18.0, v))), 1) for k, v in (s.get("mix_db") or {}).items()
                   if k in ("drums", "bass", "other", "vocals")}
    s["kit_db"] = {str(k): round(float(max(-30.0, min(0.0, v))), 1) for k, v in (s.get("kit_db") or {}).items()}
    s["level_ref_db"] = round(float(s["level_ref_db"]), 1) if s.get("level_ref_db") is not None else None
    s["master_db"] = round(float(max(-24.0, min(24.0, s.get("master_db") or 0.0))), 1)
    s["fx"] = bool(s.get("fx", True))
    p = s.get("pad")
    s["pad"] = ({"file": str(p["file"]), "base_midi": int(p.get("base_midi", 60)), "seconds": float(p.get("seconds", 0.0))}
                if isinstance(p, dict) and p.get("file") else None)
    s["motifs"] = [{"steps": [int(x) for x in m["steps"]], "degrees": [int(x) for x in m["degrees"]], "contour": m.get("contour", "flat"),
                    "count": int(m.get("count", 1)), "name": str(m.get("name", ""))} for m in (s.get("motifs") or []) if m.get("steps")]
    s["bass_cells"] = [{"steps": [int(x) for x in c["steps"]], "degrees": [int(x) for x in c["degrees"]], "count": int(c.get("count", 1))}
                       for c in (s.get("bass_cells") or []) if c.get("steps")]
    for e in s["sections"]:
        for k in ("melody", "bass_line"):
            if e.get(k):
                e[k] = [[int(n[0]), int(n[1]), int(n[2]), float(n[3]), float(n[4])] for n in e[k] if len(n) >= 5]
    s["bpm_src"] = float(s["bpm_src"]) if s.get("bpm_src") else None
    s["fidelity"] = float(max(0.0, min(1.0, s.get("fidelity") or 0.0)))
    for e in s["sections"]:
        if e.get("loops"):
            e["loops"] = {k: str(v) for k, v in e["loops"].items() if v and not str(k).startswith("_")}
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
    if (not s.get("kit") and not s.get("vocals") and not s.get("bank") and not s.get("bass_bank") and not s.get("pad")
            and not (s.get("fidelity", 0.0) > 0 and any(e.get("loops") for e in s["sections"]))):
        return style
    st = copy.deepcopy(style)
    # a song's material carries its own loudness and top end: no loudness normaliser (it pushed quiet material
    # into the limiter by +8 dB) and no master shelf; the level is set by render's calibration (master_db)
    st["target_lufs"] = None
    st["master_shelf_db"] = 0.0
    kit_db = s.get("kit_db") or {}
    for slot, path in (s.get("kit") or {}).items():
        if slot in st["slots"] and os.path.exists(path):
            base = st["slots"][slot]
            # identified sounds are peak-normalised one-shots: their balance is the song's (kit_db), not the analog
            # patch gains; 0.6 leaves the drum bus headroom when several land on one step
            gain = 0.6 * 10 ** (float(kit_db[slot]) / 20.0) if slot in kit_db else float(base.get("gain", 0.8))
            st["slots"][slot] = {"voice": "sample", "file": path, "base_midi": KIT_BASE.get(slot, 60), "gain": gain,
                                 "send_reverb": float(base.get("send_reverb", 0.0)), "octave": base.get("octave", 3)}
    if s.get("fidelity", 0.0) > 0.0 and any(e.get("loops") for e in s["sections"]):
        for slot in ("loop_drums", "loop_bass", "loop_other", "loop_vox"):
            st["slots"][slot] = {"voice": "sample", "gain": 1.0, "base_midi": 60}
        for sec in st["sections"].values():
            if "*" not in sec["layers"]:
                sec["layers"] = set(sec["layers"]) | {"loop_drums", "loop_bass", "loop_other", "loop_vox"}
        if s["fidelity"] >= 0.99:
            st["master_shelf_db"] = 0.0          # the loops carry the record's own top end
            st["target_lufs"] = None             # and its own loudness
    bank = [b for b in (s.get("bank") or []) if os.path.exists(b["file"])]
    bank_keys = [b for b in (s.get("bank_keys") or []) if os.path.exists(b["file"])] or bank
    if bank:
        for slot, bk in (("keys", bank_keys), ("arp", bank), ("lead", bank)):
            if slot in st["slots"]:
                base = st["slots"][slot]
                st["slots"][slot] = {"voice": "sample", "samples": bk, "gain": float(base.get("gain", 0.4)) * 0.9,
                                     "send_reverb": float(base.get("send_reverb", 0.3)), "send_delay": float(base.get("send_delay", 0.0)),
                                     "octave": base.get("octave", 3), "decay": 0.6}
    pad = s.get("pad")
    if pad and os.path.exists(str(pad.get("file", ""))) and "pad" in st["slots"]:
        base = st["slots"]["pad"]
        st["slots"]["pad"] = {"voice": "sample", "samples": [{"file": pad["file"], "base_midi": int(pad.get("base_midi", 60))}],
                              "gain": float(base.get("gain", 0.2)), "send_reverb": float(base.get("send_reverb", 0.4)) * 0.5,
                              "octave": base.get("octave", 3), "drone": True, "loop": True}   # one note on the root, the texture itself, held
    bass_bank = [b for b in (s.get("bass_bank") or []) if os.path.exists(b["file"])]
    if bass_bank and "bass" in st["slots"]:
        base = st["slots"]["bass"]
        st["slots"]["bass"] = {"voice": "sample", "samples": bass_bank, "gain": float(base.get("gain", 0.6)) * 1.6, "octave": base.get("octave", 1),
                               "send_reverb": 0.0, "decay": 0.9}
    if s.get("vocals"):
        st["slots"]["vox"] = {"voice": "sample", "gain": 0.85, "base_midi": 60, "send_reverb": 0.2}
        for sec in st["sections"].values():
            if "*" not in sec["layers"]:
                sec["layers"] = set(sec["layers"]) | {"vox"}
    mix = s.get("mix_db") or {}
    if any(abs(float(v)) > 0.05 for v in mix.values()):
        # per-stem trims (dB) measured against the source's own stems: the mix identified
        for slot, patch in st["slots"].items():
            stem = STEM_OF.get(slot)
            if stem in mix and isinstance(patch, dict) and "gain" in patch:
                patch["gain"] = float(patch["gain"]) * 10 ** (float(mix[stem]) / 20.0)
    return st


STEM_OF = {"kick": "drums", "snare": "drums", "hat": "drums", "ohat": "drums", "shaker": "drums", "perc": "drums", "rim": "drums",
           "tom": "drums", "ride": "drums", "bass": "bass", "pad": "other", "keys": "other", "lead": "other", "arp": "other",
           "melody": "other", "vox": "vocals"}


def make_composer(script: dict, arc_fn=None):
    from lib.gen.composer import Composer
    s = normalize(script)
    c = Composer(s["style"], bpm=s["bpm"], key=s["key"], seed=s["seed"], arc_fn=arc_fn)
    c.load_script(s)
    return c


def level_trims(audio, script: dict, bpm: float) -> list:
    """Per-section dB to add so the recreation's section levels (relative
    to its own mean) match the script's "level" targets (the source's)."""
    from lib.gen.analysis.ingest import features_on_grid, DB_PER_UNIT
    s = normalize(script)
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1) if np.ndim(audio) == 2 else np.asarray(audio, dtype=np.float32)
    feats = features_on_grid(mono, float(bpm), 0.0)
    db = np.array([f["energy_db"] for f in feats])
    if len(db) < 4:
        return [0.0 for _ in s["sections"]]
    song = float(db.mean())
    out = []
    b = 0
    for e in s["sections"]:
        seg = db[b:b + e["bars"]]
        b += e["bars"]
        if e.get("level") is None or len(seg) == 0:
            out.append(0.0)
            continue
        have = DB_PER_UNIT * (float(seg.mean()) - song)
        t = float(np.clip(float(e["level"]) - have, -9.0, 9.0))
        out.append(round(t, 1) if abs(t) >= 0.3 else 0.0)
    return out


def render(script: dict, out_path: str | None = None, seconds: float | None = None, seed: int | None = None,
           progress=None, calibrate: bool | None = None, stems: bool = False):
    """Play the script offline. Returns (audio (n,2) float32, composer).
    seconds defaults to the script's own length (plus a bar of tail).
    calibrate: when the sections carry a "level" (the analyser writes the
    source's section levels) and no "trim_db" yet, a first pass measures
    the recreation's own section levels and the difference goes into
    trim_db for the pass that is returned (composer.script carries the
    trims, so a caller can save them and skip the pass next time)."""
    s = normalize(script)
    if seed is not None:
        s["seed"] = int(seed)
    if calibrate is None:
        calibrate = (any(e.get("level") is not None for e in s["sections"]) and not any(e.get("trim_db") is not None for e in s["sections"])
                     and s.get("fidelity", 0.0) < 0.99)
    if calibrate:
        first, c0 = _render_pass(s, None, seconds, progress=(lambda p: progress(0.5 * p)) if progress else None)
        trims = level_trims(first, s, c0.bpm)
        for e, t in zip(s["sections"], trims):
            e["trim_db"] = t
        if s.get("level_ref_db") is not None:
            s["master_db"] = master_makeup(first, float(s["level_ref_db"]), float(s.get("master_db") or 0.0))
        return _render_pass(s, out_path, seconds, progress=(lambda p: progress(0.5 + 0.5 * p)) if progress else None, stems=stems)
    return _render_pass(s, out_path, seconds, progress=progress, stems=stems)


def master_makeup(audio, ref_db: float, current_db: float = 0.0, limit_into_db: float = 2.0) -> float:
    """The master_db that brings a render's RMS to ref_db (the source's),
    capped so its peaks go at most limit_into_db over full scale (the
    lookahead limiter takes that much cleanly; more is the crushed sound)."""
    m = np.asarray(audio, dtype=np.float32).mean(axis=1) if np.ndim(audio) == 2 else np.asarray(audio, dtype=np.float32)
    rms = 20.0 * np.log10(float(np.sqrt(np.mean(m ** 2))) + 1e-9)
    peak = 20.0 * np.log10(float(np.abs(audio).max()) + 1e-9)
    want = ref_db - rms
    room = limit_into_db - peak
    return round(float(max(-24.0, min(24.0, current_db + min(want, room)))), 1)


def _render_pass(s: dict, out_path, seconds, progress=None, stems: bool = False):
    """stems=True (with out_path) also writes the rack's four buses as
    <out>_drums/_bass/_other/_vocals.wav - exact stems of the recreation."""
    from lib.gen.synth import SynthRack
    c = make_composer(s)
    total = float(seconds) if seconds else total_seconds(s, c.bpm) + 4 * 60.0 / c.bpm
    full_loops = s.get("fidelity", 0.0) >= 0.99 and any(e.get("loops") for e in s["sections"])
    master = (1.0 if full_loops else 0.8) * 10 ** (float(s.get("master_db") or 0.0) / 20.0)     # loops carry the record's level
    rack = SynthRack(apply_material(c.style, s), c.bpm, seed=s["seed"], master=master)
    rack.warm_up()
    n_total = int(total * RATE)
    for p in c.phrases_until(n_total):
        rack.schedule(p.events)
    out = []
    done = 0
    writers = {}
    if stems and out_path:
        import soundfile as sf
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        root = os.path.splitext(out_path)[0]
        writers = {n: sf.SoundFile(f"{root}_{n}.wav", "w", samplerate=RATE, channels=2, subtype="PCM_16")
                   for n in ("drums", "bass", "other", "vocals")}
        gain = rack.master
        rack.capture = lambda name, block: writers[name].write(np.clip(block * gain, -1, 1))
    try:
        while rack.clock < n_total:
            out.append(rack.render(2048))
            done += 2048
            if progress is not None and done % (2048 * 64) == 0:
                progress(min(1.0, done / n_total))
    finally:
        rack.capture = None
        for w in writers.values():
            w.close()
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
            out.append((bar, "chords", [chord_str(c) for c in e["chords"]]))   # (script-only: the action list has no chord lever)
        if e.get("hook"):
            out.append((bar, "hook", e["hook"].get("name", "hook")))   # (script-only)
        if e.get("drums_phrases"):
            for p, tpl in enumerate(e["drums_phrases"]):
                if p * 4 >= e["bars"]:
                    break
                d = {k: [st for st, _ in tpl.get(k) or []] for k in ("kick", "snare", "hat")}
                if tpl.get("fill"):
                    d["fill"] = {k: [st for st, _ in v] for k, v in tpl["fill"].items()}
                out.append((bar + 4 * p, "drums", d))                   # (script-only: the beat, per phrase)
        elif e.get("drums"):
            out.append((bar, "drums", {k: [st for st, _ in v] for k, v in e["drums"].items()}))   # (script-only: the beat)
        if e.get("dyn"):
            out.append((bar, "dyn", [round(float(x), 1) for x in e["dyn"][: e["bars"]]]))       # (script-only: dB per bar -> gain lane)
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
            ch = [chord_str(x) for x in e["chords"]]
            bits.append("chords " + " ".join(ch[:8]) + (f" ..{len(ch)}" if len(ch) > 8 else ""))
        if e.get("key") or e.get("bpm"):
            bits.append(f"-> {e.get('key', '')} {e.get('bpm', '')}".strip())
        if e.get("hook"):
            bits.append("hook")
        if e.get("drums"):
            bits.append("beat " + "/".join(f"{k}:{''.join('x' if any(st == i for st, _ in v) else '.' for i in range(16))}" for k, v in e["drums"].items() if v))
        if e.get("drums_phrases"):
            n_fill = sum(1 for p in e["drums_phrases"] if p.get("fill"))
            bits.append(f"{len(e['drums_phrases'])} phrase templates" + (f", {n_fill} fills" if n_fill else ""))
        if e.get("dyn"):
            d = e["dyn"][: e["bars"]]
            bits.append(f"dyn {min(d):+.1f}..{max(d):+.1f} dB")
        lines.append("  " + "  ".join(bits))
        bar += e["bars"]
    if s.get("kit"):
        lines.append(f"  kit: {len(s['kit'])} identified sounds: " + ", ".join(f"{k}={os.path.basename(v)}" for k, v in s["kit"].items()))
    if s.get("pad"):
        lines.append(f"  pad: the song's sustained texture ({s['pad'].get('seconds', 0)} s at midi {s['pad'].get('base_midi')})")
    if s.get("bank_keys"):
        lines.append(f"  keys: a second plucked instrument ({len(s['bank_keys'])} note samples)")
    if s.get("vocals"):
        lines.append(f"  vocals: {len(s['vocals'])} phrases placed (first at bar {s['vocals'][0]['bar']})")
    if s.get("bank"):
        lines.append(f"  bank: {len(s['bank'])} melodic tones (keys/arp play the song's own sound)")
    if any(e.get("bass") for e in s["sections"]):
        lines.append("  bass: cells transcribed from the bass stem")
    if s.get("motifs"):
        lines.append(f"  motifs: {len(s['motifs'])} melody cells from the song seed the motif memory (top: {s['motifs'][0]['name']}); "
                     f"lead/keys/arp play {len(s.get('bank') or [])} of its note samples")
    if s.get("bass_cells"):
        lines.append(f"  bass: {len(s['bass_cells'])} cells from the bass stem feed the bass generator"
                     + (f", through {len(s['bass_bank'])} bass note samples" if s.get("bass_bank") else ""))
    n_mel = sum(len(e.get("melody") or []) for e in s["sections"])
    if n_mel:
        lines.append(f"  (evidence: {n_mel} transcribed melody notes, {sum(len(e.get('bass_line') or []) for e in s['sections'])} bass notes - not replayed)")
    n_loops = sum(1 for e in s["sections"] if e.get("loops"))
    if n_loops:
        lines.append(f"  loops: {n_loops} sections carry source loops; fidelity {s.get('fidelity', 0.0):.2f} "
                     "(0 generator only .. 1 the loops, generator fills the rest)")
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
