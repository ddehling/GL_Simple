"""One-shot library tool for the generative rack (lib/gen/synth/oneshots.py).

    python tools/gen/oneshots.py scan media/oneshots            # write manifest.json from the .wav files in a folder
    python tools/gen/oneshots.py scan projects/fan/media/gen/oneshots
    python tools/gen/oneshots.py bootstrap [media/oneshots]     # render a starter set from the rack's own voices
    python tools/gen/oneshots.py list                           # names the styles can reference as oneshots:<name>

Scan: every *.wav becomes an entry named by its stem; a trailing _c4 /
_a1 style suffix sets base_midi (default 60 for pitched, 36 for names
starting with kick/tom, 38 snare/clap). Tags come from the leading word
of the name (kick_a -> "kick").
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_NOTE = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5, "f#": 6, "gb": 6, "g": 7,
         "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}


def _base_midi(stem):
    m = re.search(r"_([a-g][#b]?)(-?\d)$", stem.lower())
    if m:
        return 12 * (int(m.group(2)) + 1) + _NOTE[m.group(1)]
    head = stem.split("_")[0].lower()
    if head.startswith(("kick", "tom", "boom")):
        return 36
    if head.startswith(("snare", "clap", "rim")):
        return 38
    return 60


def scan(folder):
    entries = {}
    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith(".wav"):
            continue
        stem = os.path.splitext(fn)[0]
        entries[stem] = {"file": fn, "base_midi": _base_midi(stem), "tags": [stem.split("_")[0].lower()]}
    path = os.path.join(folder, "manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=1)
    print(f"wrote {path}: {len(entries)} samples")
    return entries


def bootstrap(folder):
    """Render a starter set with the analog voices so the sample slots
    have something to play on a fresh machine. Replace with real
    recordings whenever you have them - same names, same manifest."""
    import numpy as np
    import soundfile as sf
    from lib.gen import RATE
    from lib.gen.synth.voices import VOICES
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(909)
    jobs = [
        ("kick_909", "kick909", 36.0, {"decay": 0.5, "drive": 2.6, "pitch": 50.0}, {}),
        ("kick_808", "kick", 36.0, {"decay": 0.45}, {}),
        ("clap_909", "clap909", 38.0, {}, {}),
        ("snare_a", "snare", 38.0, {}, {}),
        ("hat_closed", "hat", 42.0, {"decay": 0.05}, {}),
        ("hat_open", "hat", 46.0, {"decay": 0.25}, {}),
        ("ride_a", "ride", 51.0, {"decay": 0.9}, {}),
        ("shaker_a", "shaker", 70.0, {}, {}),
        ("tom_lo_a2", "tom", 45.0, {"decay": 0.35}, {}),
        ("tom_hi_d3", "tom", 50.0, {"decay": 0.3}, {}),
        ("rim_a", "rim", 37.0, {}, {}),
        ("impact_a", "fx", 36.0, {}, {"kind": "impact"}),
        ("riser_2s", "fx", 45.0, {}, {"kind": "riser"}),
        ("sweep_2s", "fx", 60.0, {}, {"kind": "sweep"}),
    ]
    for name, voice, pitch, patch, params in jobs:
        dur = int(2.0 * RATE) if "2s" in name else int(0.3 * RATE)
        buf = VOICES[voice]().render(pitch, 0.9, dur, patch, params, rng)
        if buf.ndim == 1:
            buf = np.stack([buf, buf], axis=1)
        pk = float(np.abs(buf).max()) or 1.0
        sf.write(os.path.join(folder, name + ".wav"), np.clip(buf / pk * 0.9, -1, 1), RATE, subtype="PCM_16")
    return scan(folder)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "bootstrap", "list"])
    ap.add_argument("folder", nargs="?", default=os.path.join("media", "oneshots"))
    args = ap.parse_args()
    if args.cmd == "scan":
        scan(args.folder)
    elif args.cmd == "bootstrap":
        bootstrap(args.folder)
    else:
        from lib.gen.synth import oneshots
        for folder, man in oneshots.manifests():
            print(folder)
            for k, v in man.items():
                print(f"  oneshots:{k:20s} {v.get('file')}  base_midi={v.get('base_midi')}  tags={','.join(v.get('tags') or [])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
