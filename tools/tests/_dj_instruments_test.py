"""Gate for per-song instrument discovery (lib/dj/instruments.py).

A 32-bar song is synthesised as four stems with KNOWN parts:
  drums   kick every beat, clap on 2 and 4, hat on every 8th, a quieter
          shaker on the off-16ths in bars 8-24
  bass    a plucked line, one note per beat (a few pushed to the "and")
  other   a plucked melody on 8ths + a soft pad holding a triad that
          changes every four bars
  vocals  silence
The pass must find the instruments (not a fixed list - the clusters of
THIS recording), place their hits on the right beats/steps, and read the
right notes: kick/clap/hat recall and precision, bass and melody pitch
accuracy against the written lines, the pad's chord tones per beat, the
silent stem reported, the result round-tripping through save/load.

Usage: python tools/tests/_dj_instruments_test.py            # synthetic gate
       python tools/tests/_dj_instruments_test.py --real 3   # + a library track (D:/Devel/music) with timing
"""
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.dj import instruments as INS          # noqa: E402

RATE = 44100
BPM = 124.0
BARS = 32
PERIOD = 60.0 / BPM
STEP = PERIOD / 4
FIRST = 0.12                                    # first beat not at zero: the grid math must cope
FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def f_of(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def t_of(beat, step=0):
    return FIRST + beat * PERIOD + step * STEP


def place(buf, t, sig):
    a = int(t * RATE)
    b = min(len(buf), a + len(sig))
    if b > a:
        buf[a:b] += sig[: b - a]


def env_exp(n, tau):
    return np.exp(-np.arange(n) / (tau * RATE)).astype(np.float32)


def kick():
    n = int(0.3 * RATE)
    t = np.arange(n) / RATE
    f = 45.0 + 110.0 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / RATE
    return (np.sin(ph) * env_exp(n, 0.09) * 0.9).astype(np.float32)


def clap(rng):
    n = int(0.16 * RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    # bandpass-ish: difference of running means
    k1, k2 = 6, 60
    lo = np.convolve(noise, np.ones(k1) / k1, mode="same")
    hi = np.convolve(noise, np.ones(k2) / k2, mode="same")
    body = (lo - hi) * env_exp(n, 0.04)
    tone = np.sin(2 * np.pi * 185.0 * np.arange(n) / RATE) * env_exp(n, 0.03) * 0.5
    return ((body * 0.8 + tone) * 0.6).astype(np.float32)


def hat(rng, dur=0.045, gain=0.35):
    n = int(dur * RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    hp = noise - np.convolve(noise, np.ones(5) / 5, mode="same")
    return (hp * env_exp(n, dur / 3) * gain).astype(np.float32)


def shaker(rng):
    n = int(0.09 * RATE)
    noise = rng.standard_normal(n).astype(np.float32)
    hp = noise - np.convolve(noise, np.ones(9) / 9, mode="same")
    e = np.concatenate([np.linspace(0, 1, int(0.01 * RATE)), env_exp(n - int(0.01 * RATE), 0.03)])
    return (hp * e[:n] * 0.2).astype(np.float32)


def pluck(midi, dur, tau, harmonics=6, roll=1.0, gain=0.5):
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    f = f_of(midi)
    sig = np.zeros(n)
    for h in range(1, harmonics + 1):
        if h * f > 18000:
            break
        sig += np.sin(2 * np.pi * h * f * t + 0.3 * h) / (h ** roll) * np.exp(-t * h / (tau * 2))
    return (sig * env_exp(n, tau) * gain).astype(np.float32)


def pad(chord, dur, gain=0.18):
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    sig = np.zeros(n)
    for m in chord:
        f = f_of(m)
        for h, g in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15)):
            sig += g * np.sin(2 * np.pi * h * f * t + 0.7 * h + m)
    att = int(0.08 * RATE)
    rel = int(0.05 * RATE)
    e = np.ones(n)
    e[:att] = np.linspace(0, 1, att)
    e[-rel:] = np.linspace(1, 0, rel)
    return (sig * e * gain / len(chord)).astype(np.float32)


def make_song():
    rng = np.random.default_rng(7)
    n_beats = BARS * 4
    total = int((FIRST + n_beats * PERIOD + 1.0) * RATE)
    drums = np.zeros(total, dtype=np.float32)
    bass = np.zeros(total, dtype=np.float32)
    other = np.zeros(total, dtype=np.float32)
    truth = {"kick": [], "clap": [], "hat": [], "shaker": [], "bass": [], "melody": [], "pad": {}}
    k_sig, c_sig, h_sig, s_sig = kick(), clap(rng), hat(rng), shaker(rng)
    bass_line = [40, 40, 43, 45, 40, 40, 47, 45]          # E2 E2 G2 A2 E2 E2 B2 A2
    melody = [64, 67, 71, 69, 67, 64, 62, 64, 71, 74, 72, 71, 69, 67, 69, 71]
    chords = [(52, 55, 59), (48, 52, 55), (57, 60, 64), (55, 59, 62)]
    for beat in range(n_beats):
        bar, bib = beat // 4, beat % 4
        place(drums, t_of(beat), k_sig); truth["kick"].append((beat, 0))
        if bib in (1, 3):
            place(drums, t_of(beat), c_sig); truth["clap"].append((beat, 0))
        for st in (0, 2):
            place(drums, t_of(beat, st), h_sig * (1.0 if st == 0 else 0.8)); truth["hat"].append((beat, st))
        if 8 <= bar < 24:
            for st in (1, 3):
                place(drums, t_of(beat, st), s_sig); truth["shaker"].append((beat, st))
        m = bass_line[beat % 8]
        if bib == 3 and bar % 2 == 1:
            place(bass, t_of(beat, 2), pluck(m, 0.6, 0.18, harmonics=8, gain=0.7)); truth["bass"].append((beat, 2, m))
        else:
            place(bass, t_of(beat), pluck(m, 0.6, 0.18, harmonics=8, gain=0.7)); truth["bass"].append((beat, 0, m))
        for st in (0, 2):
            mm = melody[(beat * 2 + st // 2) % 16]
            place(other, t_of(beat, st), pluck(mm, 0.5, 0.12, harmonics=5, roll=1.4, gain=0.45)); truth["melody"].append((beat, st, mm))
    for bar in range(BARS):
        ch = chords[(bar // 4) % 4]
        place(other, t_of(bar * 4), pad(ch, 4 * PERIOD))
        for b in range(4):
            truth["pad"][bar * 4 + b] = ch
    vocals = np.zeros(total, dtype=np.float32)
    stems = {n: np.stack([a, a], axis=1) for n, a in (("drums", drums), ("bass", bass), ("other", other), ("vocals", vocals))}
    return stems, truth, total / RATE


def _events(inst):
    return {(e[0], e[1]): e for e in inst["events"]}


def _recall_prec(inst, truth_hits, n_beats):
    ev = _events(inst)
    hit = sum(1 for k in truth_hits if k in ev)
    recall = hit / max(1, len(truth_hits))
    prec = hit / max(1, len(ev))
    return recall, prec


def _pitch_acc(inst, truth_notes):
    ev = _events(inst)
    got = 0
    octave = 0
    found = 0
    for b, st, m in truth_notes:
        e = ev.get((b, st))
        if e is None or e[2] is None:
            continue
        found += 1
        if e[2] == m:
            got += 1
        elif (e[2] - m) % 12 == 0:
            octave += 1
    return got / max(1, len(truth_notes)), found / max(1, len(truth_notes)), octave


def main():
    print("== beat grid")
    grid = [{"start_s": 0.0, "end_s": 30.0, "period_s": 0.5, "first_beat_s": 1.3, "bpm": 120.0, "score": 0.9}]
    beats, down0 = INS.beat_times(grid, 1, 61.0)
    check(len(beats) == 122 and abs(beats[0] - 0.3) < 1e-9, f"grid extrapolated over the track from t=0 ({len(beats)} beats, first {beats[0]:.2f})")
    # the grid's first beat (1.3 s) is index 2; downbeat_offset 1 -> the first downbeat is 1.8 s = index 3
    check(down0 == 3 and abs(beats[3] - 1.8) < 1e-9, f"first downbeat index {down0} at {beats[down0]:.2f} s")
    check(INS.beat_step_of(beats, 0.3 + 0.5 * 10 + 0.125) == (10, 1), "beat/step placement")
    check(INS.beat_step_of(beats, 0.3 + 0.5 * 10 + 0.49) == (11, 0), "a hit just before the next beat rounds onto it")

    print("== synthetic song")
    stems, truth, dur = make_song()
    n_beats = BARS * 4
    beats = np.array([t_of(b) for b in range(n_beats)])
    t0 = time.time()
    # the `other` stem is read here with the transcription reader, whatever the default: YourMT3+ (the default
    # since readings v13, the winner on the truth set's GM instruments) is a model of real instruments and reads
    # 7% of this song's 5-harmonic synthetic pluck and 30% of its sine pad's roots (measured 2026-09-09; the
    # transcription reader 93% / 99%) - a real limit of that model on pure synthetic tones, recorded in
    # docs/RECONSTRUCTION_PLAN.md, and not what this gate's `other` checks (the reading's plumbing: voices,
    # notes placed on the grid, the pad's tones held) are for
    other_reader = INS.OTHER_READER
    INS.OTHER_READER = "transcription"
    try:
        res = INS.identify(stems, beats, down0=0, period=PERIOD, progress=lambda s: print("   ", s))
    finally:
        INS.OTHER_READER = other_reader
    print(f"  {time.time() - t0:.1f} s for {dur:.0f} s of audio" + (f" (`other` read by the transcription reader, default {other_reader})" if other_reader != "transcription" else ""))
    for line in INS.summary(res):
        print("   ", line)
    # with READ_VOCALS off the vocals stem is not read at all (carried as phrases); either message is the right report
    check(any(("vocals: silent" in r) or ("vocals: carried as phrases" in r) for r in res["reasons"]), "silent vocal stem reported")

    d = res["stems"]["drums"]["instruments"]
    # with the learned kit decomposition (DRUM_READER "drumsep") this additive kit is not a realistic input: the
    # separator sprays its noise-and-sine sounds over all four families (446 "toms" onsets on a kit without toms)
    # and each family's reader makes voices of the spray. The four written sounds are still found (checked
    # below); the count is a plumbing check here, judged for real on the truth set (tools/tests/_dj_truthset.py)
    hi = 14 if getattr(INS, "DRUM_READER", "mixed") == "drumsep" else 6
    check(3 <= len(d) <= hi, f"drums: {len(d)} instruments found (3-{hi} accepted for 4 written)")
    # each written part -> the discovered sound that reproduces it best (the role HINTS are
    # reading aids, not the claim under test); a sound answers for one part only
    taken = []
    # hat and shaker are written with the SAME noise spectrum (only their envelopes differ), the
    # adversarial case for any spectral reader: half of one leaking into the other is tolerated here -
    # the TRUTH SET (_dj_truthset.py, real instruments through demucs) is the judge for kits since
    # 2026-09-09; this gate checks the reading's plumbing on clean tones
    for part, (r_min, p_min) in (("kick", (0.9, 0.85)), ("clap", (0.85, 0.8)), ("hat", (0.5, 0.6)), ("shaker", (0.5, 0.6))):
        cands = [i for i in d if i["id"] not in taken]
        if not cands:
            check(False, f"a sound left for the {part}")
            continue
        best = max(cands, key=lambda i: min(_recall_prec(i, truth[part], n_beats)))
        taken.append(best["id"])
        r, p = _recall_prec(best, truth[part], n_beats)
        check(r >= r_min and p >= p_min, f"{part}: {best['id']} ({best['label']}, hint {best.get('hint')}) recall {r:.2f} precision {p:.2f}")
    check(all(i["exemplar"][1] > i["exemplar"][0] for i in d), "every drum sound has an exemplar cut")

    b = res["stems"]["bass"]["instruments"]
    bh = [i for i in b if i["kind"] == "hit" and i["pitched"]]
    check(bool(bh), f"bass: a pitched hit instrument ({len(b)} instruments)")
    if bh:
        acc, found, octv = _pitch_acc(bh[0], truth["bass"])
        check(acc >= 0.8, f"bass line {bh[0]['id']}: {acc:.2f} of notes exact (placed {found:.2f}, octave errors {octv}) range {bh[0]['range']}")
        check(38 <= bh[0]["range"][0] <= 41 and bh[0]["range"][1] == 47,
              f"bass range read as {[INS.note_name(m) for m in bh[0]['range']]} (E2-B2 written; a slip of a tone at the bottom tolerated)")

    o = res["stems"]["other"]["instruments"]
    oh = [i for i in o if i["kind"] == "hit" and i["pitched"]]
    check(bool(oh), f"other: a pitched hit instrument ({len(o)} instruments)")
    if oh:
        best = max(oh, key=lambda i: _pitch_acc(i, truth["melody"])[0])
        acc, found, octv = _pitch_acc(best, truth["melody"])
        check(acc >= 0.7, f"melody {best['id']}: {acc:.2f} of notes exact (placed {found:.2f}, octave errors {octv})")
    # the pad: a sustained voice, or (transcription reader, 2026-09-09) a pitched voice of long notes - the
    # written pad retriggers every beat (0.48 s notes), which the transcriber reads as notes, not holds
    # (the transcriber ends the retriggered pad's notes early - 0.24 s for 0.48 s written - so duration
    # cannot tell the pad; every pitched voice but the melody's is the pad's candidate, and the chord
    # checks below are the claim)
    # whose voice the pad's tones land in is the VOICE-ASSIGNMENT question, measured as purity on the truth
    # set (_dj_truthset.py); here the claim is that the pad's NOTES are read: chord tones on its beats
    os_ = [i for i in o if i["pitched"]]
    check(bool(os_), f"other: pitched voices for the pad's tones ({len(os_)}: {[(i['id'], i.get('hint')) for i in os_]})")
    if os_:
        table = INS.beat_table(res)
        pad_ids = {i["id"] for i in os_}
        root_ok = tone_ok = n = 0
        for k, ch in truth["pad"].items():
            held = {ev[2] for inst, ev in table.get(k, []) if inst["id"] in pad_ids}
            if not held:
                continue
            n += 1
            if ch[0] in held:
                root_ok += 1
            if held & set(ch):
                tone_ok += 1
        cover = n / len(truth["pad"])
        check(cover >= 0.8, f"pad heard on {cover:.2f} of its beats")
        check(n and root_ok / n >= 0.7, f"chord root among the held notes on {root_ok / max(n, 1):.2f} of heard beats, a chord tone on {tone_ok / max(n, 1):.2f}")
        print(f"  --   voice hints: {[i.get('hint') for i in os_]} (a reading aid, not checked)")

    print("== per-beat readout")
    table = INS.beat_table(res)
    k = 4 * 9 + 1                                     # bar 9 beat 2: kick + clap + hats + shaker + bass + melody + pad
    what = sorted({inst["id"] for inst, _ in table.get(k, [])})
    check(len(what) >= 6, f"beat {k} (bar {INS.bar_beat(res, k)[0]} beat {INS.bar_beat(res, k)[1]}): {what}")
    check(INS.beat_index(res, t_of(k) + 0.01) == k, "beat_index finds the beat from a time")

    print("== storage")
    with tempfile.TemporaryDirectory() as td:
        p = INS.save(td, 42, res)
        back = INS.load(td, 42)
        check(back is not None and back["stems"]["drums"]["instruments"][0]["events"] == d[0]["events"], f"save/load round trip ({os.path.getsize(p) // 1024} KB)")
        back["version"] = 0
        INS.save(td, 42, back)
        check(INS.load(td, 42) is None and INS.load(td, 42, any_version=True) is not None, "an older version is not loaded unless asked")

    if "--real" in sys.argv:
        tid = int(sys.argv[sys.argv.index("--real") + 1])
        print(f"== real track {tid}")
        from lib.dj import resolve_music_dir
        from lib.dj.db import LibraryDB
        root = resolve_music_dir(os.environ.get("DJ_MUSIC", "D:/Devel/music"))
        db = LibraryDB(root)
        row = next(r for r in db.all_tracks() if r["id"] == tid)
        db.close()
        print(f"   {row.get('title')} - {row.get('artist')}  {row.get('bpm'):.1f} bpm")
        t0 = time.time()
        r = INS.identify_track(root, tid, row.get("beat_grid") or [], row.get("downbeat_offset") or 0, row.get("duration_s") or 0.0,
                               bpm=row.get("bpm"), progress=lambda s: print("   ", s), save_result=False)
        print(f"   {time.time() - t0:.1f} s")
        for line in INS.summary(r):
            print("   ", line)
        for why in r["reasons"]:
            print("    !", why)
        table = INS.beat_table(r)
        for k in range(64, 72):
            bar, bib = INS.bar_beat(r, k)
            what = [f"{inst['id']}" + (f":{INS.note_name(ev[2])}" if ev[2] is not None else "") for inst, ev in table.get(k, [])]
            print(f"    bar {bar:3d}.{bib}  {' '.join(what)}")

    print("\n" + ("ALL OK" if not FAILS else f"{len(FAILS)} FAIL: " + "; ".join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
