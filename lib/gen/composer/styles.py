"""Style presets: parameter bundles, not code. A style says which slots
exist, what patch each slot uses, how sections are shaped, how dense each
layer is at a given energy, and the harmonic grammar.

Section layer masks are the ONE lever that makes form audible: a "break"
is a section where the kick is masked, a "drop" is where everything is
in. Energy scales density inside the mask.

Patches: "voice" picks the class (lib/gen/synth/voices.py); every other
key is a knob. Pitched voices accept any name in _Subtractive.TUNABLE
(detunes, a/d/s/r, drift, filter_lfo, drive, hp ...), so two styles can
share a class and sound like different instruments. "layers" stacks more
patches under the same notes (sub + top bass, body + click kick), each
with its own gain and hp/lp crossover. "gain" is a mix decision on top
of the rack's auto gain staging (rack.SLOT_TARGET_DB): whichever voice
sits in a slot, gain 1.0 means the same loudness. Sends: send_delay,
send_reverb, send_chorus.

"drums": four | broken | breakbeat | halftime (rhythm.py); "halftime_in"
forces halftime in those sections. "feel" is the groove template: per
slot, a swing multiplier on the style's swing and a push (seconds,
+ = late). "harmony": borrow / sus probabilities, pedal_in / slow_in
sections. "auto": per-section overrides of the mix automation program
(composer.AUTO). "target_lufs": the loudness the rack holds the style at.
"reverb_decay": the FDN loop gain (0.7 tight .. 0.92 hall)."""
import copy

# section -> set of slots allowed. "*" = all slots of the style.
_CLUB_SECTIONS = {
    "intro":  {"bars": (8, 16), "layers": {"kick", "hat", "perc", "pad", "shaker"}, "energy": 0.35},
    "groove": {"bars": (16, 32), "layers": {"kick", "snare", "hat", "ohat", "perc", "shaker", "tom", "rim", "bass", "pad", "keys"}, "energy": 0.6},
    "build":  {"bars": (8, 8),  "layers": {"kick", "snare", "hat", "ohat", "perc", "shaker", "tom", "rim", "bass", "lead", "arp", "pad"}, "energy": 0.8},
    "drop":   {"bars": (16, 32), "layers": {"*"}, "energy": 1.0},
    "break":  {"bars": (8, 16), "layers": {"hat", "shaker", "pad", "keys", "lead", "arp"}, "energy": 0.4},
    "outro":  {"bars": (8, 16), "layers": {"kick", "hat", "pad"}, "energy": 0.3},
}
# The form grammar: which section may follow which (weights).
_CLUB_FORM = {
    "intro":  [("groove", 1.0)],
    "groove": [("build", 0.55), ("break", 0.3), ("groove", 0.15)],
    "build":  [("drop", 1.0)],
    "drop":   [("break", 0.5), ("groove", 0.4), ("drop", 0.1)],
    "break":  [("build", 0.6), ("groove", 0.4)],
    "outro":  [("outro", 1.0)],
}

_AMBIENT_SECTIONS = {
    "intro": {"bars": (8, 16), "layers": {"pad"}, "energy": 0.2},
    "flow":  {"bars": (16, 32), "layers": {"pad", "keys", "arp", "perc", "shaker"}, "energy": 0.4},
    "swell": {"bars": (8, 16), "layers": {"pad", "keys", "arp", "lead", "perc", "shaker", "bass"}, "energy": 0.7},
    "calm":  {"bars": (8, 16), "layers": {"pad", "keys"}, "energy": 0.25},
    "outro": {"bars": (8, 16), "layers": {"pad"}, "energy": 0.15},
}
_AMBIENT_FORM = {
    "intro": [("flow", 1.0)],
    "flow":  [("swell", 0.5), ("calm", 0.3), ("flow", 0.2)],
    "swell": [("calm", 0.6), ("flow", 0.4)],
    "calm":  [("flow", 0.7), ("swell", 0.3)],
    "outro": [("outro", 1.0)],
}

_CLUB_FEEL = {
    "hat":    {"swing": 1.2, "push": 0.004},
    "shaker": {"swing": 1.4, "push": 0.006},
    "ohat":   {"swing": 1.0, "push": 0.005},
    "snare":  {"swing": 0.0, "push": -0.002},
    "bass":   {"swing": 0.6, "push": 0.003},
    "keys":   {"swing": 0.8, "push": 0.002},
    "arp":    {"swing": 0.9, "push": 0.0},
}
_FX = {"voice": "fx", "gain": 0.5, "send_reverb": 0.35}

STYLES = {
    "groove": {
        "label": "club groove (house / techno)",
        "bpm": (120.0, 126.0),
        "mode": "minor",
        "swing": 0.08,                 # 0 straight .. 0.33 full shuffle (16ths)
        "steps_per_bar": 16,
        "drums": "four",
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [               # scale degrees (0-based), one per bar
            [0, 0, 5, 6], [0, 5, 2, 6], [0, 3, 5, 6], [0, 0, 0, 0],
            [0, 6, 5, 6], [0, 2, 5, 3], [5, 6, 0, 0],
        ],
        "progression_hold": (2, 4),    # phrases a progression persists
        "harmony": {"borrow": 0.25, "sus": 0.25, "pedal_in": ["break"], "slow_in": ["break", "intro"]},
        "feel": _CLUB_FEEL,
        "target_lufs": -14.0, "reverb_decay": 0.8,
        # hosted instruments, used when the plugin is present on this machine (else the analog patch)
        "vst": {
            "keys": {"plugin": "vst:dexed", "program": "-ANALOG 1-", "gain": 0.3, "tail": 1.5, "send_chorus": 0.3},
            "lead": {"plugin": "vst:dexed", "program": "SAW EM UP ", "gain": 0.55, "tail": 1.5, "vel_curve": 1.3},
            "pad":  {"plugin": "vst:dexed", "program": "Slow3D Pad", "gain": 0.22, "tail": 3.0},
        },
        "slots": {
            "kick":   {"voice": "kick909", "gain": 1.01, "decay": 0.5, "drive": 2.6, "pitch": 50.0,
                       "layers": [{"voice": "hat", "gain": 0.18, "decay": 0.012, "hp": 3500.0}]},      # body + click
            "snare":  {"voice": "clap909", "gain": 0.64},
            "hat":    {"voice": "hat", "gain": 0.36, "decay": 0.045},
            "ohat":   {"voice": "hat", "gain": 0.28, "decay": 0.22},
            "shaker": {"voice": "shaker", "gain": 0.37},
            "ride":   {"voice": "ride", "gain": 0.17, "decay": 0.8, "send_reverb": 0.15},
            "tom":    {"voice": "tom", "gain": 0.6, "decay": 0.3, "send_reverb": 0.2},
            "rim":    {"voice": "rim", "gain": 0.3, "send_delay": 0.25},
            "perc":   {"voice": "perc", "gain": 0.5, "send_delay": 0.2},
            "bass":   {"voice": "bass", "gain": 0.6, "cutoff": 900.0, "res": 0.45, "octave": 1, "drive": 1.7,
                       "layers": [{"voice": "pluck", "gain": 0.3, "hp": 260.0, "cutoff": 1800.0, "detunes": (-5.0, 5.0), "stereo": False}]},  # sub + top
            "lead":   {"voice": "lead", "gain": 0.7, "cutoff": 2600.0, "res": 0.4, "octave": 4, "send_delay": 0.35, "send_reverb": 0.2},
            "pad":    {"voice": "pad", "gain": 0.19, "cutoff": 1400.0, "octave": 3, "send_reverb": 0.55, "send_chorus": 0.5},
            "arp":    {"voice": "pluck", "gain": 0.44, "cutoff": 3200.0, "octave": 4, "send_delay": 0.45, "send_reverb": 0.25},
            "keys":   {"voice": "keys", "gain": 0.27, "cutoff": 2200.0, "octave": 3, "send_reverb": 0.4, "send_chorus": 0.35, "fluid_program": 4},
            "fx":     _FX,
        },
        "density": {"hat": 1.0, "shaker": 0.85, "ride": 0.9, "perc": 0.5, "tom": 0.6, "rim": 0.3, "bass": 0.8,
                    "lead": 0.45, "arp": 0.9, "keys": 0.35},
    },
    "techno": {
        "label": "techno (driving, dark)",
        "bpm": (128.0, 134.0),
        "mode": "phrygian",
        "swing": 0.02,
        "steps_per_bar": 16,
        "drums": "four",
        "perc_poly": True,
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [[0, 0, 0, 0], [0, 0, 5, 5], [0, 1, 0, 0], [0, 0, 6, 5]],
        "progression_hold": (3, 6),
        "harmony": {"borrow": 0.1, "sus": 0.15, "pedal_in": ["break", "groove"], "slow_in": ["intro", "break", "groove"]},
        "feel": {"hat": {"swing": 1.0, "push": 0.002}, "shaker": {"swing": 1.0, "push": 0.003},
                 "snare": {"swing": 0.0, "push": -0.001}, "bass": {"swing": 0.0, "push": 0.002}},
        "target_lufs": -12.5, "reverb_decay": 0.85,
        "auto": {"drop": {"duck": (0.7, 0)}, "break": {"lp": (2500.0, 4), "verb": (2.0, 2)}},
        "slots": {
            "kick":   {"voice": "kick909", "gain": 1.05, "decay": 0.42, "drive": 3.2, "pitch": 48.0, "sweep": 0.05},
            "snare":  {"voice": "clap909", "gain": 0.5, "tail": 0.16},
            "rim":    {"voice": "rim", "gain": 0.35, "send_delay": 0.4},
            "hat":    {"voice": "hat", "gain": 0.34, "decay": 0.035},
            "ohat":   {"voice": "hat", "gain": 0.26, "decay": 0.18},
            "shaker": {"voice": "shaker", "gain": 0.3, "hz": 8000.0},
            "ride":   {"voice": "ride", "gain": 0.2, "decay": 1.1, "send_reverb": 0.2},
            "tom":    {"voice": "tom", "gain": 0.5, "decay": 0.25},
            "perc":   {"voice": "perc", "gain": 0.55, "send_delay": 0.45},
            "bass":   {"voice": "bass", "gain": 0.62, "cutoff": 700.0, "res": 0.5, "octave": 1, "drive": 2.4, "sub": 0.7,
                       "layers": [{"voice": "lead", "gain": 0.25, "hp": 200.0, "cutoff": 1200.0, "detunes": (-9.0, 9.0), "stereo": False, "drive": 2.0}]},
            "lead":   {"voice": "lead", "gain": 0.55, "cutoff": 1800.0, "res": 0.55, "octave": 3, "pulse": 1.0, "drive": 2.2,
                       "send_delay": 0.5, "send_reverb": 0.3, "vibrato": 0.0},
            "pad":    {"voice": "pad", "gain": 0.2, "cutoff": 800.0, "octave": 3, "send_reverb": 0.7, "send_chorus": 0.4,
                       "filter_lfo": 0.5, "filter_lfo_hz": 0.1},
            "arp":    {"voice": "pluck", "gain": 0.4, "cutoff": 2600.0, "octave": 4, "send_delay": 0.55, "send_reverb": 0.3, "d": 0.12},
            "keys":   {"voice": "fm", "gain": 0.24, "octave": 4, "send_reverb": 0.5, "send_delay": 0.3,
                       "fm_ratio": 7.0, "fm_index": 1.2, "fm_decay": 0.15, "d": 0.2, "s": 0.0, "r": 0.2},
            "fx":     _FX,
        },
        "density": {"hat": 1.0, "shaker": 0.7, "ride": 0.8, "perc": 0.7, "tom": 0.5, "rim": 0.5, "bass": 0.9,
                    "lead": 0.35, "arp": 0.8, "keys": 0.3},
    },
    "trance": {
        "label": "trance (uplifting, supersaw)",
        "bpm": (136.0, 140.0),
        "mode": "minor",
        "swing": 0.0,
        "steps_per_bar": 16,
        "drums": "four",
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [[0, 5, 3, 6], [0, 3, 5, 6], [5, 6, 0, 0], [0, 6, 3, 6], [0, 5, 6, 4]],
        "progression_hold": (2, 4),
        "harmony": {"borrow": 0.2, "sus": 0.35, "pedal_in": ["intro"], "slow_in": ["intro", "outro"]},
        "feel": {"hat": {"swing": 1.0, "push": 0.002}, "snare": {"swing": 0.0, "push": -0.002},
                 "bass": {"swing": 0.0, "push": 0.002}, "arp": {"swing": 0.0, "push": 0.0}},
        "target_lufs": -12.5, "reverb_decay": 0.9,
        "auto": {"build": {"verb": (1.6, 2)}, "drop": {"duck": (0.65, 0), "verb": (1.1, 0)}},
        "slots": {
            "kick":   {"voice": "kick909", "gain": 1.0, "decay": 0.46, "drive": 2.4, "pitch": 52.0},
            "snare":  {"voice": "clap909", "gain": 0.6},
            "hat":    {"voice": "hat", "gain": 0.34, "decay": 0.04},
            "ohat":   {"voice": "hat", "gain": 0.3, "decay": 0.24},
            "shaker": {"voice": "shaker", "gain": 0.3},
            "ride":   {"voice": "ride", "gain": 0.16, "decay": 0.9, "send_reverb": 0.2},
            "tom":    {"voice": "tom", "gain": 0.55, "decay": 0.32, "send_reverb": 0.3},
            "perc":   {"voice": "perc", "gain": 0.4, "send_delay": 0.35},
            "bass":   {"voice": "bass", "gain": 0.6, "cutoff": 1100.0, "res": 0.35, "octave": 1, "drive": 1.5, "d": 0.08, "s": 0.4,
                       "layers": [{"voice": "pluck", "gain": 0.35, "hp": 250.0, "cutoff": 2400.0, "stereo": False}]},
            "lead":   {"voice": "supersaw", "gain": 0.6, "cutoff": 3500.0, "res": 0.2, "octave": 4, "send_delay": 0.4, "send_reverb": 0.35,
                       "a": 0.01, "d": 0.3, "s": 0.7, "r": 0.4},
            "pad":    {"voice": "supersaw", "gain": 0.22, "cutoff": 2200.0, "octave": 3, "send_reverb": 0.7, "send_chorus": 0.4,
                       "a": 0.8, "d": 1.0, "s": 0.85, "r": 1.2},
            "arp":    {"voice": "pluck", "gain": 0.5, "cutoff": 3600.0, "octave": 4, "send_delay": 0.55, "send_reverb": 0.3},
            "keys":   {"voice": "keys", "gain": 0.26, "cutoff": 2600.0, "octave": 3, "send_reverb": 0.5, "send_chorus": 0.4},
            "fx":     _FX,
        },
        "density": {"hat": 1.0, "shaker": 0.6, "ride": 0.8, "perc": 0.4, "tom": 0.5, "bass": 1.0,
                    "lead": 0.5, "arp": 1.0, "keys": 0.3},
    },
    "dnb": {
        "label": "drum and bass (breaks, sub)",
        "bpm": (170.0, 176.0),
        "mode": "minor",
        "swing": 0.05,
        "steps_per_bar": 16,
        "drums": "breakbeat",
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [[0, 0, 5, 5], [0, 3, 0, 6], [0, 0, 0, 0], [0, 6, 5, 6]],
        "progression_hold": (3, 5),
        "harmony": {"borrow": 0.15, "sus": 0.2, "pedal_in": ["break", "groove"], "slow_in": ["intro", "break"]},
        "feel": {"hat": {"swing": 1.0, "push": 0.003}, "snare": {"swing": 0.0, "push": -0.002},
                 "bass": {"swing": 0.0, "push": 0.002}, "shaker": {"swing": 1.2, "push": 0.004}},
        "target_lufs": -12.5, "reverb_decay": 0.78,
        "auto": {"drop": {"duck": (0.55, 0)}},
        "slots": {
            "kick":   {"voice": "kick909", "gain": 0.95, "decay": 0.3, "drive": 2.8, "pitch": 55.0, "sweep": 0.035},
            "snare":  {"voice": "snare", "gain": 0.75,
                       "layers": [{"voice": "clap909", "gain": 0.35, "hp": 1200.0}]},
            "hat":    {"voice": "hat", "gain": 0.32, "decay": 0.03},
            "ohat":   {"voice": "hat", "gain": 0.24, "decay": 0.16},
            "shaker": {"voice": "shaker", "gain": 0.28},
            "ride":   {"voice": "ride", "gain": 0.15, "decay": 0.7},
            "tom":    {"voice": "tom", "gain": 0.5, "decay": 0.22},
            "perc":   {"voice": "perc", "gain": 0.4, "send_delay": 0.3},
            "bass":   {"voice": "bass", "gain": 0.65, "cutoff": 420.0, "res": 0.3, "octave": 1, "drive": 2.0, "sub": 1.0, "a": 0.01, "r": 0.15,
                       "layers": [{"voice": "lead", "gain": 0.3, "hp": 140.0, "cutoff": 900.0, "detunes": (-14.0, 14.0), "stereo": False,
                                   "drive": 2.6, "vibrato": 0.0, "filter_lfo": 0.3, "filter_lfo_hz": 0.5}]},   # sub + reese
            "lead":   {"voice": "fm", "gain": 0.45, "octave": 4, "send_delay": 0.45, "send_reverb": 0.4,
                       "fm_ratio": 2.0, "fm_index": 2.0, "fm_decay": 0.25, "d": 0.3, "s": 0.3, "r": 0.3},
            "pad":    {"voice": "pad", "gain": 0.2, "cutoff": 1100.0, "octave": 3, "send_reverb": 0.7, "send_chorus": 0.4},
            "arp":    {"voice": "pluck", "gain": 0.4, "cutoff": 3000.0, "octave": 4, "send_delay": 0.5, "send_reverb": 0.3},
            "keys":   {"voice": "keys", "gain": 0.24, "cutoff": 2000.0, "octave": 3, "send_reverb": 0.5},
            "fx":     _FX,
        },
        "density": {"hat": 0.9, "shaker": 0.5, "ride": 0.7, "perc": 0.4, "tom": 0.4, "bass": 0.9,
                    "lead": 0.4, "arp": 0.7, "keys": 0.3},
    },
    "hiphop": {
        "label": "hip-hop (halftime, dusty)",
        "bpm": (86.0, 94.0),
        "mode": "dorian",
        "swing": 0.22,
        "steps_per_bar": 16,
        "drums": "halftime",
        "halftime_in": ["intro", "groove", "build", "drop", "break", "outro"],
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [[0, 3, 0, 6], [0, 0, 3, 4], [2, 5, 0, 0], [0, 5, 3, 6]],
        "progression_hold": (2, 4),
        "harmony": {"borrow": 0.3, "sus": 0.3, "pedal_in": ["break"], "slow_in": ["intro", "break"]},
        "feel": {"hat": {"swing": 1.1, "push": 0.008}, "shaker": {"swing": 1.2, "push": 0.01},
                 "snare": {"swing": 0.5, "push": 0.006}, "bass": {"swing": 0.8, "push": 0.006},
                 "keys": {"swing": 1.0, "push": 0.006}, "rim": {"swing": 1.0, "push": 0.004}},
        "target_lufs": -13.5, "reverb_decay": 0.75,
        "auto": {"drop": {"duck": (0.4, 0)}, "break": {"lp": (4500.0, 4)}},
        "slots": {
            "kick":   {"voice": "kick", "gain": 1.0, "decay": 0.6, "pitch": 42.0, "sweep": 0.03,
                       "layers": [{"voice": "sample", "gain": 0.45, "file": "oneshots:kick_909", "base_midi": 36, "lp": 1500.0}]},
            "snare":  {"voice": "snare", "gain": 0.7,
                       "layers": [{"voice": "rim", "gain": 0.4}]},
            "rim":    {"voice": "rim", "gain": 0.3, "send_delay": 0.35},
            "hat":    {"voice": "hat", "gain": 0.34, "decay": 0.05},
            "ohat":   {"voice": "hat", "gain": 0.24, "decay": 0.3},
            "shaker": {"voice": "shaker", "gain": 0.28, "hz": 5600.0},
            "tom":    {"voice": "tom", "gain": 0.45, "decay": 0.4, "send_reverb": 0.3},
            "perc":   {"voice": "perc", "gain": 0.4, "send_delay": 0.3},
            "bass":   {"voice": "bass", "gain": 0.6, "cutoff": 350.0, "res": 0.2, "octave": 1, "drive": 1.6, "sub": 0.9, "glide": 0.08},
            "lead":   {"voice": "ks", "gain": 0.5, "octave": 4, "send_delay": 0.4, "send_reverb": 0.35,
                       "decay": 0.996, "brightness": 0.45, "body": 0.3},
            "pad":    {"voice": "pad", "gain": 0.2, "cutoff": 900.0, "octave": 3, "send_reverb": 0.6, "send_chorus": 0.5,
                       "detunes": (-8.0, 8.0), "a": 0.3, "r": 0.8},
            "arp":    {"voice": "ks", "gain": 0.4, "octave": 4, "send_delay": 0.45, "send_reverb": 0.35,
                       "decay": 0.994, "brightness": 0.6},
            "keys":   {"voice": "fm", "gain": 0.3, "octave": 3, "send_reverb": 0.45, "send_chorus": 0.3,
                       "fm_ratio": 2.0, "fm_index": 1.4, "fm_decay": 0.5, "d": 0.6, "s": 0.15, "r": 0.35},
            "fx":     _FX,
        },
        "density": {"hat": 0.8, "shaker": 0.5, "rim": 0.4, "perc": 0.35, "tom": 0.3, "bass": 0.6,
                    "lead": 0.35, "arp": 0.5, "keys": 0.5},
    },
    "downtempo": {
        "label": "downtempo / broken beat",
        "bpm": (88.0, 98.0),
        "mode": "dorian",
        "swing": 0.18,
        "steps_per_bar": 16,
        "drums": "broken",
        "halftime_in": ["break"],
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [[0, 3, 0, 6], [0, 0, 3, 4], [0, 5, 3, 6], [2, 5, 0, 0]],
        "progression_hold": (2, 3),
        "harmony": {"borrow": 0.3, "sus": 0.3, "pedal_in": ["break", "intro"], "slow_in": ["break"]},
        "feel": {
            "hat":    {"swing": 1.1, "push": 0.006},
            "shaker": {"swing": 1.25, "push": 0.008},
            "rim":    {"swing": 1.0, "push": 0.003},
            "snare":  {"swing": 0.3, "push": -0.004},
            "bass":   {"swing": 0.7, "push": 0.005},
            "keys":   {"swing": 0.9, "push": 0.004},
        },
        "target_lufs": -15.0, "reverb_decay": 0.82,
        "slots": {
            "kick":   {"voice": "kick", "gain": 0.95, "decay": 0.28},
            "snare":  {"voice": "snare", "gain": 0.47},
            "rim":    {"voice": "rim", "gain": 0.45, "send_delay": 0.3},
            "hat":    {"voice": "hat", "gain": 0.28, "decay": 0.06},
            "ohat":   {"voice": "hat", "gain": 0.19, "decay": 0.3},
            "shaker": {"voice": "shaker", "gain": 0.3, "hz": 6200.0},
            "tom":    {"voice": "tom", "gain": 0.55, "decay": 0.4, "send_reverb": 0.3},
            "perc":   {"voice": "perc", "gain": 0.45, "send_delay": 0.25},
            "bass":   {"voice": "bass", "gain": 0.58, "cutoff": 500.0, "res": 0.3, "octave": 1, "drive": 2.2, "sub": 0.8},
            "lead":   {"voice": "lead", "gain": 0.6, "cutoff": 1800.0, "res": 0.3, "octave": 4, "send_delay": 0.45, "send_reverb": 0.35,
                       "pulse": 0.8, "vibrato": 0.004},
            "pad":    {"voice": "pad", "gain": 0.27, "cutoff": 1000.0, "octave": 3, "send_reverb": 0.6, "send_chorus": 0.45},
            "arp":    {"voice": "pluck", "gain": 0.41, "cutoff": 2400.0, "octave": 4, "send_delay": 0.5, "send_reverb": 0.3},
            "keys":   {"voice": "fm", "gain": 0.34, "octave": 3, "send_reverb": 0.45, "send_chorus": 0.3, "fm_ratio": 2.0, "fm_index": 1.6,
                       "fm_decay": 0.4, "d": 0.5, "s": 0.15, "r": 0.3, "fluid_program": 0},
            "fx":     {"voice": "fx", "gain": 0.41, "send_reverb": 0.4},
        },
        "density": {"hat": 0.7, "shaker": 0.6, "rim": 0.5, "perc": 0.4, "tom": 0.4, "bass": 0.55,
                    "lead": 0.4, "arp": 0.6, "keys": 0.5},
    },
    "ambient": {
        "label": "ambient / generative drift",
        "bpm": (60.0, 72.0),
        "mode": "lydian",
        "swing": 0.0,
        "steps_per_bar": 16,
        "drums": "broken",
        "sections": _AMBIENT_SECTIONS, "form": _AMBIENT_FORM, "first": "intro",
        "progressions": [[0, 0, 3, 3], [0, 4, 3, 1], [0, 0, 0, 0], [3, 0, 1, 4]],
        "progression_hold": (3, 6),
        "harmony": {"borrow": 0.15, "sus": 0.4, "pedal_in": ["intro", "calm", "outro"], "slow_in": ["intro", "flow", "calm", "outro"]},
        "feel": {
            "shaker": {"swing": 1.0, "push": 0.01},
            "arp":    {"swing": 1.0, "push": 0.006},
            "keys":   {"swing": 1.0, "push": 0.008},
        },
        "target_lufs": -20.0, "reverb_decay": 0.92,
        "slots": {
            "perc":   {"voice": "perc", "gain": 0.25, "send_delay": 0.4, "send_reverb": 0.4},
            "shaker": {"voice": "shaker", "gain": 0.16, "hz": 5200.0, "send_reverb": 0.5},
            "bass":   {"voice": "bass", "gain": 0.4, "cutoff": 300.0, "res": 0.15, "octave": 1, "drive": 1.0, "a": 0.08, "r": 0.5},
            # the same PadVoice class as groove, a different instrument: wider
            # detune, slower breath, deeper filter drift, longer attack
            "pad":    {"voice": "pad", "gain": 0.37, "cutoff": 900.0, "octave": 3, "send_reverb": 0.8, "send_chorus": 0.6,
                       "detunes": (-18.0, -9.0, -3.0, 3.0, 9.0, 18.0), "a": 1.6, "r": 2.5, "fa": 3.0, "fd": 4.0,
                       "drift": 7.0, "filter_lfo": 0.5, "filter_lfo_hz": 0.07},
            "lead":   {"voice": "fm", "gain": 0.5, "octave": 5, "send_delay": 0.6, "send_reverb": 0.6,
                       "fm_ratio": 3.5, "fm_index": 2.4, "fm_decay": 0.6, "a": 0.01, "d": 1.2, "s": 0.0, "r": 1.5},
            "arp":    {"voice": "ks", "gain": 0.4, "octave": 4, "send_delay": 0.6, "send_reverb": 0.6,
                       "decay": 0.997, "brightness": 0.35, "body": 0.4},
            "keys":   {"voice": "keys", "gain": 0.29, "cutoff": 1500.0, "octave": 4, "send_reverb": 0.7, "send_chorus": 0.5,
                       "d": 0.9, "r": 0.6, "tremolo": 0.4, "fluid_program": 11},
        },
        "density": {"perc": 0.2, "shaker": 0.3, "bass": 0.3, "lead": 0.3, "arp": 0.5, "keys": 0.35},
    },
}


def get_style(name: str) -> dict:
    if name not in STYLES:
        raise KeyError(f"unknown style {name!r}; have {sorted(STYLES)}")
    st = copy.deepcopy(STYLES[name])
    try:
        from lib.gen.analysis import learn
        st = learn.apply(name, st)          # data-derived adjustments from ingested songs (GEN_LEARNED=0 to skip)
    except Exception:
        pass
    return st
