"""Style presets: parameter bundles, not code. A style says which slots
exist, what patch each slot uses, how sections are shaped, how dense each
layer is at a given energy, and the harmonic grammar.

Section layer masks are the ONE lever that makes form audible: a "break"
is a section where the kick is masked, a "drop" is where everything is
in. Energy scales density inside the mask.
"""
import copy

# section -> set of slots allowed. "*" = all slots of the style.
_CLUB_SECTIONS = {
    "intro":  {"bars": (8, 16), "layers": {"kick", "hat", "perc", "pad"}, "energy": 0.35},
    "groove": {"bars": (16, 32), "layers": {"kick", "snare", "hat", "ohat", "perc", "bass", "pad", "keys"}, "energy": 0.6},
    "build":  {"bars": (8, 8),  "layers": {"kick", "snare", "hat", "ohat", "perc", "bass", "lead", "arp", "pad"}, "energy": 0.8},
    "drop":   {"bars": (16, 32), "layers": {"*"}, "energy": 1.0},
    "break":  {"bars": (8, 16), "layers": {"hat", "pad", "keys", "lead", "arp"}, "energy": 0.4},
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
    "flow":  {"bars": (16, 32), "layers": {"pad", "keys", "arp", "perc"}, "energy": 0.4},
    "swell": {"bars": (8, 16), "layers": {"pad", "keys", "arp", "lead", "perc", "bass"}, "energy": 0.7},
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

STYLES = {
    "groove": {
        "label": "club groove (house / techno)",
        "bpm": (120.0, 126.0),
        "mode": "minor",
        "swing": 0.08,                 # 0 straight .. 0.33 full shuffle (16ths)
        "steps_per_bar": 16,
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [               # scale degrees (0-based), one per bar
            [0, 0, 5, 6], [0, 5, 2, 6], [0, 3, 5, 6], [0, 0, 0, 0],
            [0, 6, 5, 6], [0, 2, 5, 3], [5, 6, 0, 0],
        ],
        "progression_hold": (2, 4),    # phrases a progression persists
        "slots": {
            "kick":  {"voice": "kick", "gain": 1.0},
            "snare": {"voice": "clap", "gain": 0.55},
            "hat":   {"voice": "hat", "gain": 0.32, "decay": 0.045},
            "ohat":  {"voice": "hat", "gain": 0.26, "decay": 0.22},
            "perc":  {"voice": "perc", "gain": 0.3},
            "bass":  {"voice": "bass", "gain": 0.7, "cutoff": 900.0, "res": 0.35, "octave": 1},
            "lead":  {"voice": "lead", "gain": 0.3, "cutoff": 2600.0, "res": 0.25, "octave": 4, "send_delay": 0.35, "send_reverb": 0.2},
            "pad":   {"voice": "pad", "gain": 0.25, "cutoff": 1400.0, "octave": 3, "send_reverb": 0.55},
            "arp":   {"voice": "pluck", "gain": 0.28, "cutoff": 3200.0, "octave": 4, "send_delay": 0.45, "send_reverb": 0.25},
            "keys":  {"voice": "pluck", "gain": 0.24, "cutoff": 2200.0, "octave": 3, "send_reverb": 0.4, "fluid_program": 4},
        },
        "density": {"hat": 1.0, "perc": 0.5, "bass": 0.8, "lead": 0.45, "arp": 0.9, "keys": 0.35},
    },
    "downtempo": {
        "label": "downtempo / broken beat",
        "bpm": (88.0, 98.0),
        "mode": "dorian",
        "swing": 0.18,
        "steps_per_bar": 16,
        "sections": _CLUB_SECTIONS, "form": _CLUB_FORM, "first": "intro",
        "progressions": [[0, 3, 0, 6], [0, 0, 3, 4], [0, 5, 3, 6], [2, 5, 0, 0]],
        "progression_hold": (2, 3),
        "slots": {
            "kick":  {"voice": "kick", "gain": 0.9, "decay": 0.28},
            "snare": {"voice": "snare", "gain": 0.5},
            "hat":   {"voice": "hat", "gain": 0.28, "decay": 0.06},
            "ohat":  {"voice": "hat", "gain": 0.2, "decay": 0.3},
            "perc":  {"voice": "perc", "gain": 0.3},
            "bass":  {"voice": "bass", "gain": 0.65, "cutoff": 500.0, "res": 0.2, "octave": 1},
            "lead":  {"voice": "lead", "gain": 0.25, "cutoff": 1800.0, "res": 0.2, "octave": 4, "send_delay": 0.45, "send_reverb": 0.35},
            "pad":   {"voice": "pad", "gain": 0.28, "cutoff": 1000.0, "octave": 3, "send_reverb": 0.6},
            "arp":   {"voice": "pluck", "gain": 0.22, "cutoff": 2400.0, "octave": 4, "send_delay": 0.5, "send_reverb": 0.3},
            "keys":  {"voice": "pluck", "gain": 0.3, "cutoff": 1600.0, "octave": 3, "send_reverb": 0.45, "fluid_program": 0},
        },
        "density": {"hat": 0.7, "perc": 0.4, "bass": 0.55, "lead": 0.4, "arp": 0.6, "keys": 0.5},
    },
    "ambient": {
        "label": "ambient / generative drift",
        "bpm": (60.0, 72.0),
        "mode": "lydian",
        "swing": 0.0,
        "steps_per_bar": 16,
        "sections": _AMBIENT_SECTIONS, "form": _AMBIENT_FORM, "first": "intro",
        "progressions": [[0, 0, 3, 3], [0, 4, 3, 1], [0, 0, 0, 0], [3, 0, 1, 4]],
        "progression_hold": (3, 6),
        "slots": {
            "perc":  {"voice": "perc", "gain": 0.15},
            "bass":  {"voice": "bass", "gain": 0.4, "cutoff": 300.0, "res": 0.1, "octave": 1},
            "lead":  {"voice": "lead", "gain": 0.2, "cutoff": 1400.0, "res": 0.15, "octave": 5, "send_delay": 0.6, "send_reverb": 0.6},
            "pad":   {"voice": "pad", "gain": 0.32, "cutoff": 900.0, "octave": 3, "send_reverb": 0.8},
            "arp":   {"voice": "pluck", "gain": 0.18, "cutoff": 2000.0, "octave": 4, "send_delay": 0.6, "send_reverb": 0.5},
            "keys":  {"voice": "pluck", "gain": 0.25, "cutoff": 1500.0, "octave": 4, "send_reverb": 0.7, "fluid_program": 11},
        },
        "density": {"perc": 0.2, "bass": 0.3, "lead": 0.3, "arp": 0.5, "keys": 0.35},
    },
}


def get_style(name: str) -> dict:
    if name not in STYLES:
        raise KeyError(f"unknown style {name!r}; have {sorted(STYLES)}")
    return copy.deepcopy(STYLES[name])
