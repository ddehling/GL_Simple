"""Real-instrument truth songs from Slakh (BabySlakh: the first 20 Slakh2100 tracks,
professionally sampled instruments, per-stem audio + aligned MIDI, CC BY 4.0,
16 kHz) in the truth set's own folder format, so every reading change is also
judged on material that behaves like a record, not a GM SoundFont.

    python tools/tests/_dj_truthset_slakh.py build [Track00001 ...]   # logs/slakh/babyslakh_16k/<track> -> logs/truthset/slakh_<track>/
    python tools/tests/_dj_truthset_slakh.py retruth [Track00001 ...] # rewrite truth.json only (no separation)
    python tools/tests/_dj_truthset.py eval slakh_Track00001 ...       # then the usual gates

Per track: the stems' MIDI gives the notes (program, is_drum from metadata.yaml),
the stems are summed into our four stems (drums / bass / other / vocals by GM
class), upsampled to 44.1 kHz, the mix separated with the library's demucs model,
truth.json written with bpm / first beat from the MIDI tempo map. Parts are named
"<inst_class>_<stem id>". Tracks are cut to at most MAX_S seconds.

The MIDI is not the truth of PITCH by itself: a sampled patch may sound an octave
from the written note (every BabySlakh electric bass sounds 12 semitones below its
MIDI - 2026-09-09, measured on the true stems), and a reader judged against the
MIDI would score 0.00 for hearing right. Each BASS part's octave is measured on
its own audio (pyin over its longest notes) and the notes shifted by the whole
octaves found ("octave_shift" in the part); a part that does not measure cleanly
keeps the MIDI as written. `other` parts keep the MIDI: measured there, the same
test shifted organs and strings by a sub-octave pyin heard and the truth went
wrong (see _collect).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RATE = 44100
SRC = os.path.join("logs", "slakh", "babyslakh_16k")
OUT = os.path.join("logs", "truthset")
MAX_S = 100.0
DRUM_PITCHES = {36: "kick", 35: "kick", 38: "snare", 40: "snare", 37: "rim", 39: "clap", 42: "hat", 44: "hat", 46: "ohat",
                49: "crash", 57: "crash", 51: "ride", 59: "ride", 53: "ride", 45: "tom_lo", 41: "tom_lo", 43: "tom_lo",
                47: "tom_hi", 48: "tom_hi", 50: "tom_hi", 70: "shaker", 69: "shaker"}
OCTAVE_NOTES = 40              # the longest notes a part's octave is measured on
OCTAVE_AGREE = 0.8             # ...and the share of them that must agree on the same whole-octave offset


def _stem_of(program, is_drum, inst_class):
    if is_drum:
        return "drums"
    if 32 <= int(program) <= 39 or (inst_class or "").lower() == "bass":
        return "bass"
    return "other"


def _midi_notes(path):
    """[(t_s, dur_s, midi, vel)] from a MIDI file with its own tempo map (pretty_midi)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    notes = []
    for ins in pm.instruments:
        for n in ins.notes:
            notes.append([round(float(n.start), 4), round(float(n.end - n.start), 4), int(n.pitch), round(n.velocity / 127.0, 3)])
    bpm = float(np.median(pm.get_tempo_changes()[1])) if len(pm.get_tempo_changes()[1]) else 120.0
    beats = pm.get_beats()
    first = float(beats[0]) if len(beats) else 0.0
    downs = pm.get_downbeats()
    first_down = float(downs[0]) if len(downs) else first
    return notes, bpm, first_down


def _resample(a, sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(RATE, sr)
    return resample_poly(a, RATE // g, sr // g, axis=0).astype(np.float32)


def octave_offset(a, notes):
    """Whole octaves (semitones, a multiple of 12) the part's audio sounds from its MIDI: pyin over the
    sustained part of its longest notes, the rounded semitone offset per note, kept when OCTAVE_AGREE of
    the measured notes agree on one octave. 0 when nothing measures or the offset is not an octave."""
    import librosa
    offs = []
    for t, d, m, _v in sorted(notes, key=lambda n: -n[1])[:OCTAVE_NOTES]:
        lo, hi = int((t + 0.05) * RATE), int((t + min(d, 0.6)) * RATE)
        if hi - lo < 4096 or hi > len(a):
            continue
        fmin = max(25.0, librosa.midi_to_hz(m - 26))
        fmax = min(RATE / 4.0, librosa.midi_to_hz(m + 14))
        f0, vf, _p = librosa.pyin(a[lo:hi], fmin=fmin, fmax=fmax, sr=RATE, frame_length=4096)
        f0 = f0[vf & np.isfinite(f0)]
        if len(f0):
            offs.append(int(round(float(librosa.hz_to_midi(float(np.median(f0)))))) - int(m))
    if len(offs) < 8:
        return 0
    vals, cnt = np.unique(offs, return_counts=True)
    best = int(vals[np.argmax(cnt)])
    if cnt.max() / len(offs) < OCTAVE_AGREE or best % 12 != 0:
        return 0
    return best


def _collect(tr):
    """One BabySlakh track -> (parts, stems_audio {stem: stereo}, mix stereo before gain, bpm, first beat)."""
    import yaml
    import soundfile as sf
    folder = os.path.join(SRC, tr)
    with open(os.path.join(folder, "metadata.yaml"), encoding="utf-8") as fh:
        meta = yaml.safe_load(fh)
    parts, stems_audio, bpm, first = {}, {}, None, None
    total = None
    for sid, sm in sorted(meta["stems"].items()):
        # BabySlakh's metadata says audio_rendered: false for stems whose audio IS there: the files decide
        wav = os.path.join(folder, "stems", f"{sid}.wav")
        mid = os.path.join(folder, "MIDI", f"{sid}.mid")
        if not (os.path.exists(wav) and os.path.exists(mid)):
            continue
        a, sr = sf.read(wav, dtype="float32")
        if a.ndim == 2:
            a = a.mean(axis=1)
        a = _resample(a, sr) if sr != RATE else a
        n_max = int(MAX_S * RATE)
        a = a[:n_max]
        notes, t_bpm, t_first = _midi_notes(mid)
        notes = [n for n in notes if n[0] < MAX_S]
        if bpm is None:
            bpm, first = t_bpm, t_first
        is_drum = bool(sm.get("is_drum", False))
        stem = _stem_of(sm.get("program_num", 0), is_drum, sm.get("inst_class"))
        pname = f"{(sm.get('inst_class') or 'part').lower().replace(' ', '_')}_{sid}"
        shift = 0
        # BASS parts only: there the offset is systematic and confirmed (the reader and the true stem agree on
        # -12 on every note). On `other` the same measurement shifted organs and strings by -12 (pyin lands on a
        # 16' drawbar / a sub-octave of a rich tone) and the truth went wrong: exact F1 0.74 -> 0.55 and 0.73 ->
        # 0.49 on the two tracks with shifted parts while octave-blind stayed - YourMT3 had agreed with the MIDI
        if stem == "bass" and notes:
            shift = octave_offset(a, notes)
            if shift:
                notes = [[n[0], n[1], n[2] + shift, n[3]] for n in notes]
        parts[pname] = {"stem": stem, "program": int(sm.get("program_num", 0)), "is_drum": is_drum,
                        "midi_program_name": sm.get("midi_program_name"), "octave_shift": shift, "notes": notes, "_audio": a}
        stereo = np.stack([a, a], axis=1)
        if total is None:
            total = np.zeros_like(stereo)
        if len(stereo) > len(total):
            total = np.concatenate([total, np.zeros((len(stereo) - len(total), 2), dtype=np.float32)])
        total[: len(stereo)] += stereo
        acc = stems_audio.setdefault(stem, np.zeros_like(total))
        if len(acc) < len(total):
            acc = np.concatenate([acc, np.zeros((len(total) - len(acc), 2), dtype=np.float32)])
        acc[: len(stereo)] += stereo
        stems_audio[stem] = acc
    return parts, stems_audio, total, bpm, first


def _truth(tr, parts, bpm, first, g):
    from tools.tests import _dj_truthset as TS
    parts = {k: {kk: vv for kk, vv in v.items() if kk != "_audio"} for k, v in parts.items()}
    # drum parts -> ONE "kit" part in the GM truth set's format (notes at the truth set's drum pitches, so the
    # eval scores per sound by pitch and the render gate finds the kit under the name it knows)
    kit = []
    for pname, pd in list(parts.items()):
        if pd["is_drum"]:
            for n in pd["notes"]:
                snd = DRUM_PITCHES.get(n[2])
                if snd:
                    kit.append([n[0], n[1], TS.DRUM.get(snd, n[2]), n[3]])
            del parts[pname]
    if kit:
        parts["kit"] = {"stem": "drums", "program": 0, "is_drum": True, "notes": sorted(kit)}
    return {"name": f"slakh_{tr}", "bpm": bpm, "first_beat_s": first, "bars": int(MAX_S * bpm / 60 / 4), "swing": 0.0, "gain": g,
            "source": "BabySlakh (CC BY 4.0)", "parts": parts}


def _write_kit_audio(out, parts, _g):
    """true_kit.wav: the drum parts' audio summed (raw, like the other true_<part> files - the gate applies the
    mix gain from truth.json)."""
    import soundfile as sf
    drums = [pd["_audio"] for pd in parts.values() if pd["is_drum"]]
    if not drums:
        return
    n = max(len(a) for a in drums)
    kit = np.zeros(n, dtype=np.float32)
    for a in drums:
        kit[: len(a)] += a
    sf.write(os.path.join(out, "true_kit.wav"), np.stack([kit, kit], axis=1), RATE, subtype="PCM_16")


def _report(tr, truth, out):
    shifts = {k: v["octave_shift"] for k, v in truth["parts"].items() if v.get("octave_shift")}
    print(f"{tr}: {len(truth['parts'])} parts ({sum(len(p['notes']) for p in truth['parts'].values())} notes), bpm {truth['bpm']:.1f}, "
          f"first beat {truth['first_beat_s']:.2f}, octave shifts {shifts or 'none'} -> {out}")


def build(tracks=None):
    import soundfile as sf
    from tools.tests import _dj_truthset as TS
    tracks = tracks or sorted(d for d in os.listdir(SRC) if d.startswith("Track"))
    for tr in tracks:
        parts, stems_audio, total, bpm, first = _collect(tr)
        if total is None:
            print(f"{tr}: no stems")
            continue
        out = os.path.join(OUT, f"slakh_{tr}")
        os.makedirs(os.path.join(out, "stems"), exist_ok=True)
        peak = float(np.abs(total).max()) or 1.0
        g = 10 ** (-1.0 / 20) / peak
        mix = total * g
        for pname, pd in parts.items():
            a = pd["_audio"]
            sf.write(os.path.join(out, f"true_{pname}.wav"), np.stack([a, a], axis=1), RATE, subtype="PCM_16")
        _write_kit_audio(out, parts, g)
        for st, a in stems_audio.items():
            sf.write(os.path.join(out, f"true_stem_{st}.wav"), a[: len(mix)] * g, RATE, subtype="PCM_16")
        sf.write(os.path.join(out, "mix.wav"), mix, RATE, subtype="PCM_16")
        seps = TS.separate(mix)
        for st, a in seps.items():
            sf.write(os.path.join(out, "stems", f"{st}.wav"), a, RATE, subtype="PCM_16")
        truth = _truth(tr, parts, bpm, first, g)
        with open(os.path.join(out, "truth.json"), "w", encoding="utf-8") as fh:
            json.dump(truth, fh)
        _report(tr, truth, out)


def retruth(tracks=None):
    """Rewrite truth.json for already-built tracks (the audio, stems and separation stay)."""
    tracks = tracks or sorted(d[len("slakh_"):] for d in os.listdir(OUT) if d.startswith("slakh_Track"))
    for tr in tracks:
        out = os.path.join(OUT, f"slakh_{tr}")
        path = os.path.join(out, "truth.json")
        if not os.path.exists(path):
            print(f"{tr}: not built")
            continue
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        parts, _stems, total, bpm, first = _collect(tr)
        if total is None:
            continue
        truth = _truth(tr, parts, bpm, first, old.get("gain", 1.0))
        _write_kit_audio(out, parts, old.get("gain", 1.0))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(truth, fh)
        _report(tr, truth, out)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "build":
        build(args[1:] or None)
    elif args and args[0] == "retruth":
        retruth(args[1:] or None)
    else:
        print(__doc__)
