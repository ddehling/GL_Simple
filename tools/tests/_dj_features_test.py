"""Phase A gate for the DJ subsystem: offline analysis + scanner + DB.

Synthesizes tracks with KNOWN ground truth (tempo, downbeat, key, structure)
and asserts lib/dj/features.py recovers all of it; then exercises the
incremental scanner end-to-end against a temp music folder (scan, re-scan
skips, touch one file -> only it rescans).

Usage:
    python tools/tests/_dj_features_test.py                    # ALL PASS gate
    python tools/tests/_dj_features_test.py --folder <dir>     # report on real files
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from lib.dj import features as F

RATE = F.RATE
BPM = 128.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT

# Ground-truth structure of the main synthetic track (bars):
#   intro 8 | groove 16 | build 8 | drop 16 | outro 8   = 56 bars = 105 s
INTRO_E = 8 * BAR
GROOVE_E = INTRO_E + 16 * BAR
BUILD_E = GROOVE_E + 8 * BAR
DROP_E = BUILD_E + 16 * BAR
OUTRO_E = DROP_E + 8 * BAR
TOTAL = OUTRO_E + 3.0                    # trailing silence

# A natural minor pitch material (Hz).
A1, A2, A3 = 55.0, 110.0, 220.0
C4, E4, B3, D4 = 261.63, 329.63, 246.94, 293.66


def _synth(seed=42):
    n = int(TOTAL * RATE)
    x = np.zeros(n)
    rng = np.random.RandomState(seed)

    def kick(t0, amp=0.9):
        i0 = int(t0 * RATE)
        d = int(0.10 * RATE)
        if i0 + d > n:
            return
        tl = np.arange(d) / RATE
        x[i0:i0 + d] += amp * np.sin(2 * np.pi * 52.0 * tl) * np.exp(-tl / 0.045)

    def hat(t0, amp=0.09):
        i0 = int(t0 * RATE)
        d = int(0.03 * RATE)
        if i0 + d > n:
            return
        b = rng.randn(d) * np.exp(-np.arange(d) / (0.008 * RATE))
        x[i0:i0 + d] += amp * np.diff(np.concatenate([[0.0], b]))

    def bass_note(t0, f, dur, amp):
        i0 = int(t0 * RATE)
        d = int(dur * RATE)
        if i0 + d > n:
            return
        tl = np.arange(d) / RATE
        env = np.minimum(1.0, tl / 0.01) * np.exp(-tl / (dur * 0.6))
        x[i0:i0 + d] += amp * np.sin(2 * np.pi * f * tl) * env

    def pad(t0, t1, freqs, amp):
        i0, i1 = int(t0 * RATE), min(int(t1 * RATE), n)
        tl = np.arange(i1 - i0) / RATE
        for f, a in freqs:
            x[i0:i1] += amp * a * np.sin(2 * np.pi * f * (tl + t0))

    am_chord = [(A2, 1.0), (A3, 0.7), (C4, 0.55), (E4, 0.55)]

    # Hats on 8ths from 0 (so the grid has phase through the intro too).
    t = 0.0
    while t < OUTRO_E:
        amp = 0.05 if t < INTRO_E or t >= DROP_E else 0.09
        hat(t, amp)
        t += BEAT / 2
    # Kicks on quarters through groove/build/drop, ACCENTED on bar starts
    # (that accent + the bar-start bass note IS the downbeat ground truth).
    t = INTRO_E
    k = 0
    while t < DROP_E:
        kick(t, 1.0 if k % 4 == 0 else 0.72)
        k += 1
        t += BEAT
    # Bassline: root A on the 1, fifths/thirds after - anchors key + downbeat.
    t = INTRO_E
    while t < DROP_E:
        loud = 0.5 if t >= BUILD_E + 8 * BAR - 1e-3 or t < GROOVE_E else 0.4
        bass_note(t, A1, BEAT * 0.9, loud)                 # beat 1: A root
        bass_note(t + 2 * BEAT, 82.41, BEAT * 0.8, 0.28)   # beat 3: E
        t += BAR
    # Pads: intro + groove quiet, drop bright (add high stack), outro fading.
    pad(0.0, INTRO_E, am_chord, 0.16)
    pad(INTRO_E, GROOVE_E, am_chord, 0.10)
    pad(GROOVE_E, BUILD_E, am_chord + [(B3, 0.4), (D4, 0.4)], 0.12)
    pad(BUILD_E, DROP_E, am_chord + [(E4 * 2, 0.5), (A3 * 2, 0.5)], 0.17)
    pad(DROP_E, OUTRO_E, am_chord, 0.10)
    # Build: rising noise sweep for energy slope.
    i0, i1 = int(GROOVE_E * RATE), int(BUILD_E * RATE)
    sweep = rng.randn(i1 - i0) * np.linspace(0.01, 0.16, i1 - i0)
    x[i0:i1] += sweep
    # Outro fade.
    i0, i1 = int(DROP_E * RATE), int(OUTRO_E * RATE)
    x[i0:i1] *= np.linspace(1.0, 0.25, i1 - i0)

    peak = np.max(np.abs(x))
    return (x / peak * 0.85).astype(np.float32)


def _synth_tempo_change():
    """40 s @ 128 BPM then 40 s @ 96 BPM (kick+hats only)."""
    def groove(bpm, seconds, seed):
        beat = 60.0 / bpm
        n = int(seconds * RATE)
        x = np.zeros(n)
        rng = np.random.RandomState(seed)
        t = 0.0
        while t < seconds:
            i0 = int(t * RATE)
            d = int(0.10 * RATE)
            if i0 + d <= n:
                tl = np.arange(d) / RATE
                x[i0:i0 + d] += 0.9 * np.sin(2 * np.pi * 52 * tl) * np.exp(-tl / 0.045)
            t += beat
        t = 0.0
        while t < seconds:
            i0 = int(t * RATE)
            d = int(0.03 * RATE)
            if i0 + d <= n:
                b = rng.randn(d) * np.exp(-np.arange(d) / (0.008 * RATE))
                x[i0:i0 + d] += 0.09 * np.diff(np.concatenate([[0.0], b]))
            t += beat / 2
        tt = np.arange(n) / RATE
        return x + 0.05 * np.sin(2 * np.pi * 220 * tt)
    x = np.concatenate([groove(128.0, 40.0, 1), groove(96.0, 40.0, 2)])
    return (x / np.max(np.abs(x)) * 0.85).astype(np.float32)


failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def test_analysis():
    print("Synthetic 128-BPM A-minor track (intro/groove/build/drop/outro):\n")
    samples = _synth()
    t0 = time.time()
    r = F.analyze_samples(samples, deep=True)
    dt = time.time() - t0

    check("bpm", abs(r["bpm"] - BPM) / BPM < 0.01,
          f"bpm={r['bpm']:.2f} (want {BPM:.0f} +/- 1%), analyzed in {dt:.1f}s")
    check("bpm confidence", r["bpm_conf"] > 0.5,
          f"bpm_conf={r['bpm_conf']:.2f} (want > 0.5)")

    # Beat accuracy: grid beats vs true kick times in the groove.
    grid = r["beat_grid"]
    check("single tempo segment", len(grid) == 1,
          f"{len(grid)} grid segments (steady track -> 1)")
    g = grid[0]
    beats = []
    t = g["first_beat_s"]
    while t < OUTRO_E:
        beats.append(t)
        t += g["period_s"]
    beats = np.array(beats)
    true_beats = np.arange(INTRO_E, DROP_E - 1e-6, BEAT)
    errs = [np.min(np.abs(beats - tb)) for tb in true_beats]
    med_err = float(np.median(errs)) * 1000
    check("beat alignment", med_err <= 15.0,
          f"median beat error {med_err:.1f} ms (want <= 15)")

    # Downbeat: beats[offset::4] must land on bar starts (k * BAR).
    off = r["downbeat_offset"]
    downs = beats[off::4]
    derr = [min(abs((d % BAR)), abs(BAR - (d % BAR))) for d in downs]
    med_derr = float(np.median(derr)) * 1000
    check("downbeat", med_derr <= 40.0,
          f"offset={off}, median downbeat error {med_derr:.0f} ms "
          f"(conf={r['downbeat_conf']:.2f})")
    check("downbeat confident", r["downbeat_conf"] > 0.1,
          f"downbeat_conf={r['downbeat_conf']:.2f}")

    check("key", r["camelot"] == "8A",
          f"{r['key_name']} = {r['camelot']} conf={r['key_conf']:.2f} "
          f"(want A minor = 8A)")

    secs = r["sections"]
    check("sections found", 4 <= len(secs) <= 12,
          f"{len(secs)} sections: " +
          " ".join(f"{s['kind']}@{s['start_s']:.0f}" for s in secs))
    # The drop is a MOMENT (energy slams up into the loud section), marked as
    # a cue - not a multi-minute section label. It should land in the loud
    # region [GROOVE_E, DROP_E], where energy jumps up.
    drops = [c["time_s"] for c in r["cues"] if c.get("label") == "drop"]
    check("drop moment located",
          any(GROOVE_E - 5 <= d <= DROP_E for d in drops),
          f"drop cues at {[round(d) for d in drops]} "
          f"(loud region {GROOVE_E:.0f}-{DROP_E:.0f}s)")
    check("high-energy section is groove not 'drop'",
          any(s["kind"] == "groove" for s in secs)
          and not any(s["kind"] == "drop" for s in secs),
          f"kinds: {sorted(set(s['kind'] for s in secs))}")
    check("intro low energy", secs[0]["energy"] < 0.6,
          f"first section kind={secs[0]['kind']} energy={secs[0]['energy']}")
    # Boundary near the build->drop moment (the most audible edge).
    bounds = [s["start_s"] for s in secs[1:]]
    near = min(abs(b - BUILD_E) for b in bounds) if bounds else 99.0
    check("boundary at the drop", near <= 2 * BAR,
          f"nearest boundary {near:.1f}s from true drop start (want <= {2*BAR:.1f})")

    check("loops found", len(r["loops"]) >= 1,
          f"{len(r['loops'])} loops, best={r['loops'][0] if r['loops'] else None}")
    ins = [p for p in r["mix_points"] if p["kind"] == "in"]
    outs = [p for p in r["mix_points"] if p["kind"] == "out"]
    check("mix-in points", len(ins) >= 1 and min(p["time_s"] for p in ins) < TOTAL * 0.5,
          f"{len(ins)} in-points, first at {min((p['time_s'] for p in ins), default=-1):.0f}s")
    check("mix-out points", len(outs) >= 1 and max(p["time_s"] for p in outs) > TOTAL * 0.5,
          f"{len(outs)} out-points, last at {max((p['time_s'] for p in outs), default=-1):.0f}s")

    check("loudness gain sane", -9.0 <= r["loudness_gain_db"] <= 9.0,
          f"gain={r['loudness_gain_db']} dB")
    curve = r["energy_curve"]
    i_intro = int(INTRO_E / 2 * 2 / 2)        # ~mid-intro sample at 2 Hz
    e_intro = float(np.mean(curve[2:int(INTRO_E * 2) - 2]))
    e_drop = float(np.mean(curve[int(BUILD_E * 2) + 2:int(DROP_E * 2) - 2]))
    check("energy curve shape", e_drop > e_intro * 1.5,
          f"intro {e_intro:.2f} vs drop {e_drop:.2f} (drop must dominate)")

    lc = r.get("live_check", {})
    check("live detector agrees", lc.get("agrees", False),
          f"live_bpm={lc.get('live_bpm')} mean_conf={lc.get('mean_conf')}")
    check("mood is dancey", any(k in ("groove", "peak") and v > 0.3
                                for k, v in r["mood_hist"].items()),
          f"mood_hist={r['mood_hist']}")


def test_chroma_origin():
    print("\nChroma origin (A-origin bins -> C-origin key):\n")
    tt = np.arange(int(12.0 * RATE)) / RATE
    tone = (0.5 * np.sin(2 * np.pi * 440.0 * tt)).astype(np.float32)
    bands, chroma = F.frame_track(tone)
    dom = int(np.argmax(chroma.mean(axis=0)))
    check("A440 hits A-origin bin 0", dom == 0,
          f"dominant chroma bin={dom} (analyzer convention: bin 0 = A)")
    # A major chord (A C# E) should detect as A major = 11B after rotation.
    chord = (0.4 * np.sin(2 * np.pi * 440.0 * tt)
             + 0.3 * np.sin(2 * np.pi * 554.37 * tt)
             + 0.3 * np.sin(2 * np.pi * 659.25 * tt)).astype(np.float32)
    bands, chroma = F.frame_track(chord)
    energy = np.ones(len(chroma))
    pc, mode, camelot, conf = F.estimate_key(chroma, energy)
    check("A-major chord -> 11B", camelot == "11B" and pc == 9,
          f"pc={pc} mode={mode} camelot={camelot}")


def test_tempo_change():
    print("\nTempo-change track (128 -> 96 BPM):\n")
    r = F.analyze_samples(_synth_tempo_change(), deep=False)
    grid = r["beat_grid"]
    bpms = sorted(g["bpm"] for g in grid)
    has128 = any(abs(b - 128) / 128 < 0.02 for b in bpms)
    has96 = any(abs(b - 96) / 96 < 0.02 for b in bpms)
    check("two tempo segments", len(grid) >= 2 and has128 and has96,
          f"{len(grid)} segments, bpms={[round(b,1) for b in bpms]}")


def test_scanner():
    print("\nIncremental scanner (temp library, in-process workers):\n")
    from scipy.io import wavfile
    from lib.dj.scan import scan_library
    from lib.dj.db import LibraryDB

    tmp = tempfile.mkdtemp(prefix="gl_dj_test_")
    try:
        samples = _synth()
        short = _synth_tempo_change()[:int(45 * RATE)]
        wavfile.write(os.path.join(tmp, "track_a.wav"),
                      RATE, (samples * 32767).astype(np.int16))
        wavfile.write(os.path.join(tmp, "track_b.wav"),
                      RATE, (short * 32767).astype(np.int16))

        s1 = scan_library(tmp, workers=1)
        check("first scan analyzes all", s1["scanned"] == 2 and s1["errors"] == 0,
              f"scanned={s1['scanned']} errors={s1['errors']}")
        s2 = scan_library(tmp, workers=1)
        check("re-scan skips unchanged", s2["scanned"] == 0 and s2["skipped"] == 2,
              f"scanned={s2['scanned']} skipped={s2['skipped']}")
        # Touch one file: bump its mtime well past the 1 s tolerance.
        pa = os.path.join(tmp, "track_a.wav")
        st = os.stat(pa)
        os.utime(pa, (st.st_atime, st.st_mtime + 30))
        s3 = scan_library(tmp, workers=1)
        check("touched file rescans alone", s3["scanned"] == 1 and s3["skipped"] == 1,
              f"scanned={s3['scanned']} skipped={s3['skipped']}")

        db = LibraryDB(tmp)
        t = db.get_track(rel_path="track_a.wav")
        check("db round-trip", t is not None
              and abs(t["bpm"] - BPM) / BPM < 0.01
              and t["camelot"] == "8A"
              and isinstance(t["beat_grid"], list)
              and len(db.sections_for(t["id"])) >= 4
              and len(db.mix_points_for(t["id"])) >= 2,
              f"bpm={t['bpm'] if t else '?'} camelot={t['camelot'] if t else '?'} "
              f"sections={len(db.sections_for(t['id'])) if t else 0}")
        # Missing-file flagging.
        os.remove(os.path.join(tmp, "track_b.wav"))
        db.close()
        s4 = scan_library(tmp, workers=1)
        db = LibraryDB(tmp)
        check("missing flagged, history kept", s4["missing"] == 1
              and db.counts()["total"] == 2,
              f"missing={s4['missing']} total_rows={db.counts()['total']}")
        db.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def folder_report(folder):
    """Real-library mode: analyze N files and print a judgment table."""
    from lib.dj.features import analyze_file
    import glob
    files = sorted(sum((glob.glob(os.path.join(folder, f"*{e}"))
                        for e in (".mp3", ".wav", ".flac", ".ogg")), []))
    print(f"Report on {len(files)} files in {folder}\n")
    print(f"{'file':44s} {'bpm':>6s} {'cf':>4s} {'key':>4s} {'kf':>4s} "
          f"{'secs':>4s} {'loops':>5s} {'live':>5s} {'t':>5s}")
    print("-" * 96)
    for p in files:
        t0 = time.time()
        try:
            r = analyze_file(p, deep=True)
        except Exception as e:
            print(f"{os.path.basename(p)[:44]:44s}  ERROR {type(e).__name__}: {e}")
            continue
        lc = r.get("live_check", {})
        print(f"{os.path.basename(p)[:44]:44s} {r['bpm']:6.1f} "
              f"{r['bpm_conf']:4.2f} {r['camelot']:>4s} {r['key_conf']:4.2f} "
              f"{len(r['sections']):4d} {len(r['loops']):5d} "
              f"{'OK' if lc.get('agrees') else 'DIS':>5s} "
              f"{time.time() - t0:5.1f}")


def main():
    if "--folder" in sys.argv:
        folder_report(sys.argv[sys.argv.index("--folder") + 1])
        return
    print("DJ features/scanner offline test\n" + "=" * 40 + "\n")
    test_analysis()
    test_chroma_origin()
    test_tempo_change()
    test_scanner()
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
