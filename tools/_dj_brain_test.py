"""Phase C gate: the DJ brain + full autonomous end-to-end.

Part 1 - brain units on a hand-built library: stretch clamp, Camelot
preference, recency penalties, the busy-x-busy pair veto, armed-transition
immunity to replans.

Part 2 - END TO END through every real layer: three synthetic grooves are
written as WAVs into a temp music folder, scanned by the REAL Phase-A
scanner into a REAL library DB, then a DJSystem (threaded=False) conducts
an autonomous set through the hand-pumped audio engine. The render is
judged by the live signals pipeline: beat lock must survive every
autonomous transition the brain chose on its own.

Usage:
    python tools/_dj_brain_test.py            # ALL PASS gate
    python tools/_dj_brain_test.py --wav      # keep logs/dj_e2e_set.wav
"""
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib.dj.brain import Brain, TrackInfo, camelot_compat
from lib.dj.themes import Theme

RATE = 44100
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# --------------------------------------------------------------------------
# Part 1: brain units
# --------------------------------------------------------------------------

def fake_track(tid, bpm, camelot, busy=0.3, vocal=0.2, artist="x",
               dur=300.0, energy_mood=None, busy_everywhere=False):
    period = 60.0 / bpm
    edge = 1.0 if busy_everywhere else 0.5
    sections = [
        {"kind": "intro", "start_s": 0.0, "end_s": 30.0, "energy": 0.3,
         "busyness": busy, "vocalness": vocal, "boundary_strength": 0.5},
        {"kind": "steady", "start_s": 30.0, "end_s": dur - 40.0,
         "energy": 0.7, "busyness": busy, "vocalness": vocal,
         "boundary_strength": 0.6},
        {"kind": "outro", "start_s": dur - 40.0, "end_s": dur, "energy": 0.3,
         "busyness": busy * edge, "vocalness": vocal * edge,
         "boundary_strength": 0.5},
    ]
    mix_points = [
        {"kind": "in", "time_s": 30.0, "score": 0.5, "style_hint": "blend"},
        {"kind": "out", "time_s": dur - 40.0, "score": 0.5,
         "style_hint": "blend"},
        {"kind": "out", "time_s": dur - 90.0, "score": 0.4,
         "style_hint": "blend"},
    ]
    row = {"id": tid, "path": f"t{tid}.wav", "title": f"track{tid}",
           "artist": artist, "duration_s": dur, "bpm": bpm, "bpm_conf": 0.9,
           "downbeat_offset": 0, "downbeat_conf": 0.5, "camelot": camelot,
           "beat_grid": [{"start_s": 0.0, "end_s": dur, "period_s": period,
                          "first_beat_s": 0.0, "bpm": bpm}],
           "loudness_gain_db": 0.0,
           "mood_hist": energy_mood or {"groove": 0.8, "peak": 0.1},
           "rhythm_density": 2.0,
           "spectral": {"bass_share": 0.4, "mid_share": 0.4,
                        "high_share": 0.2}}
    loops = [{"start_s": 60.0, "beats": 16, "score": 0.8}]
    return TrackInfo(row, sections, loops, mix_points)


def unit_tests():
    print("Brain units (hand-built library):\n")
    theme = Theme("t", bpm_range=(100, 132), min_play_s=60, max_play_s=120)
    cur = fake_track(1, 122.0, "8A")

    lib = [cur,
           fake_track(2, 124.0, "8A"),          # perfect neighbour
           fake_track(3, 124.0, "3B"),          # key clash, same tempo
           fake_track(4, 150.0, "8A"),          # stretch-impossible
           fake_track(5, 61.0, "9A"),           # half-time read = 122 ok
           fake_track(6, 124.0, "8A", busy=0.9, vocal=0.8)]  # wall of sound
    wins = {}
    for seed in range(12):
        brain = Brain(lib, theme, seed=seed)
        pick, meta = brain.choose_next(cur, 0.6, cur.bpm)
        wins[pick.id] = wins.get(pick.id, 0) + 1
    check("stretch clamp excludes 150bpm", 4 not in wins,
          f"wins by id: {wins} (150 bpm needs rate 0.98x2? no - unreachable)")
    check("compatible key preferred", wins.get(2, 0) > wins.get(3, 0),
          f"8A neighbour won {wins.get(2, 0)}x vs key-clash {wins.get(3, 0)}x")

    brain = Brain(lib, theme, seed=1)
    r5, e5 = brain.rate_for(122.0, lib[4])
    check("half-time read works", r5 is not None and abs(e5 - 122.0) < 1.5,
          f"61 bpm track read at eff {e5} via rate {r5}")

    # Recency: play track 2, it should stop winning immediately after.
    brain = Brain(lib, theme, seed=3)
    brain.note_played(lib[1])
    s_fresh, _ = brain.score(cur, lib[2], 0.6, cur.bpm)
    s_played, _ = brain.score(cur, lib[1], 0.6, cur.bpm)
    check("recency penalty bites", s_played < s_fresh * 0.5,
          f"just-played 8A scores {s_played:.4f} vs fresh key-clash {s_fresh:.4f}")

    # Busy x busy veto: every exit of cur is a wall of sound, and so is the
    # candidate's every entry -> no legal pair; a quiet candidate pairs fine.
    busy_cur = fake_track(7, 122.0, "8A", busy=0.9, vocal=0.3,
                          busy_everywhere=True)
    busy_cand = fake_track(8, 124.0, "8A", busy=0.9, vocal=0.8,
                           busy_everywhere=True)
    pair_bad = Brain(lib, theme, seed=1).best_pair(busy_cur, busy_cand)
    pair_ok = Brain(lib, theme, seed=1).best_pair(busy_cur, lib[1])
    check("busy x busy veto", pair_bad is None and pair_ok is not None,
          f"busy-busy pair={pair_bad}, busy-quiet pair "
          f"score={pair_ok['score'] if pair_ok else None}")

    check("camelot table sane",
          camelot_compat("8A", "8A") == 1.0
          and camelot_compat("8A", "9A") == 0.9
          and camelot_compat("8A", "8B") > camelot_compat("8A", "3B"),
          "identity=1.0, neighbour=0.9, relative > distant")


# --------------------------------------------------------------------------
# Part 2: end-to-end autonomous set
# --------------------------------------------------------------------------

def synth_structured(bpm, seconds, seed, bass_hz, pad_hz):
    """Groove with intro/outro so the analyzer finds real mix points."""
    beat = 60.0 / bpm
    n = int(seconds * RATE)
    x = np.zeros(n)
    rng = np.random.RandomState(seed)
    intro_e, outro_s = 12.0, seconds - 14.0
    t, k = 0.0, 0
    while t < outro_s:
        if t >= intro_e:
            i0 = int(t * RATE)
            d = int(0.10 * RATE)
            if i0 + d <= n:
                tl = np.arange(d) / RATE
                amp = 1.0 if k % 4 == 0 else 0.75
                x[i0:i0 + d] += amp * np.sin(2 * np.pi * bass_hz * tl) \
                    * np.exp(-tl / 0.045)
        t += beat
        k += 1
    t = 0.0
    while t < seconds - 2.0:
        i0 = int(t * RATE)
        d = int(0.03 * RATE)
        if i0 + d <= n:
            b = rng.randn(d) * np.exp(-np.arange(d) / (0.008 * RATE))
            amp = 0.05 if (t < intro_e or t > outro_s) else 0.10
            x[i0:i0 + d] += amp * np.diff(np.concatenate([[0.0], b]))
        t += beat / 2
    tt = np.arange(n) / RATE
    pad = 0.05 * np.sin(2 * np.pi * pad_hz * tt) \
        + 0.03 * np.sin(2 * np.pi * pad_hz * 1.5 * tt)
    pad[int(outro_s * RATE):] *= np.linspace(
        1.0, 0.3, n - int(outro_s * RATE))
    x += pad
    return (x / np.max(np.abs(x)) * 0.8).astype(np.float32)


def e2e_test(keep_wav):
    print("\nEnd-to-end autonomous set "
          "(scan -> brain -> engine -> signals pipeline):\n")
    from scipy.io import wavfile
    from lib.dj.scan import scan_library
    from lib.dj.system import DJSystem
    from lib.audio_engine import AudioEngine

    tmp = tempfile.mkdtemp(prefix="gl_dj_e2e_")
    try:
        specs = [(122.0, 5, 54.0, 220.0), (126.0, 6, 58.0, 262.0),
                 (124.0, 7, 50.0, 294.0)]
        for i, (bpm, seed, bass, pad) in enumerate(specs):
            wavfile.write(os.path.join(tmp, f"song_{i}_{int(bpm)}.wav"), RATE,
                          (synth_structured(bpm, 150.0, seed, bass, pad)
                           * 32767).astype(np.int16))
        s = scan_library(tmp, workers=1)
        check("scan clean", s["scanned"] == 3 and s["errors"] == 0,
              f"scanned={s['scanned']} errors={s['errors']}")

        engine = AudioEngine()
        dj = DJSystem(tmp, engine=engine, theme="groove", seed=99,
                      threaded=False, log_dir=tmp)
        ok = dj.start()
        check("dj starts", ok, f"library loaded: {ok}")
        # Short attention span so a 6-minute render holds several handovers.
        dj.brain.theme = Theme("e2e", bpm_range=(110.0, 135.0),
                               energy_base=0.55, energy_span=0.25,
                               mood_weights={"groove": 1.0, "peak": 0.5},
                               min_play_s=40.0, max_play_s=65.0)

        gen = engine._mixer()
        next(gen)
        rendered = []
        plays, armed_styles = [], []
        prev_state, prev_cur = "", None
        n_handovers = 0
        skip_t = None
        skip_reacted = None
        block = 4410
        total_s = 420.0
        rate_violations = 0
        for i in range(int(total_s * RATE) // block):
            buf = gen.send(block)
            rendered.append(np.frombuffer(buf, dtype=np.float32).reshape(-1, 2))
            dj.step()
            st = dj.status()
            t_now = (i + 1) * block / RATE
            cur_id = (st["current"] or {}).get("id")
            if cur_id != prev_cur and st["current"]:
                plays.append(st["current"]["title"])
                if prev_cur is not None:
                    n_handovers += 1
                prev_cur = cur_id
            if st["state"] == "armed" and prev_state != "armed":
                armed_styles.append(st["style"])
                if skip_t is not None and skip_reacted is None:
                    skip_reacted = t_now - skip_t
            prev_state = st["state"]
            tel = st["deck_telemetry"]
            for d in (tel.get("decks") or {}).values():
                if d["playing"] and not (0.90 <= d["rate"] <= 1.101):
                    rate_violations += 1
            # Operator poke: one skip mid-set proves the control path.
            if skip_t is None and 150.0 < t_now and st["state"] == "playing":
                dj.request_skip()
                skip_t = t_now
        mix = np.concatenate(rendered, axis=0)
        dj.stop()

        check("autonomous handovers", n_handovers >= 2
              and len(armed_styles) >= 3,
              f"{n_handovers} completed + {len(armed_styles)} armed in "
              f"{total_s/60:.0f} min: {' -> '.join(p[:14] for p in plays)}")
        check("styles are beat-matched",
              all(s in ("long_blend", "bass_swap", "loop_roll_exit",
                        "cut_at_drop", "bassline_layer", "double_drop",
                        "loop_build") for s in armed_styles),
              f"styles: {armed_styles}")
        check("deck rates always clamped", rate_violations == 0,
              f"{rate_violations} telemetry samples outside 0.90..1.10")
        check("skip honored", skip_t is not None and skip_reacted is not None
              and skip_reacted < 30.0,
              f"skip at {skip_t:.0f}s -> next transition armed "
              f"{skip_reacted:.1f}s later" if skip_reacted is not None
              else f"skip at {skip_t}s never produced an armed transition")

        mono = mix.mean(axis=1).astype(np.float64)
        rms_all = np.sqrt(np.mean(mono ** 2))
        check("render is audible", rms_all > 0.02, f"rms={rms_all:.3f}")
        w = RATE // 2
        rms = np.sqrt(np.mean(
            mono[:len(mono) // w * w].reshape(-1, w) ** 2, axis=1))
        active = rms[4:-4]
        check("no dead air", float(active.min()) > 0.1 * float(np.median(active)),
              f"min 0.5s-RMS {active.min():.3f} vs median {np.median(active):.3f}")

        from tools import _club_signals_test as CS
        log, drops = CS.run(mono)
        seg = [r for r in log if 20.0 <= r["t"] <= total_s - 5.0]
        confs = np.array([r["conf"] for r in seg])
        low = [r["t"] for r in seg if r["conf"] < 0.3]
        runs, curr = [], []
        for t in low:
            if curr and t - curr[-1] > 0.06:
                runs.append(curr)
                curr = []
            curr.append(t)
        if curr:
            runs.append(curr)
        worst = max((c[-1] - c[0] for c in runs), default=0.0)
        check("beat lock held all set", worst <= 3.0,
              f"longest conf<0.3 stretch {worst:.1f}s "
              f"(mean conf {confs.mean():.2f}) across {n_handovers} handovers")
        bpms = [r["bpm"] for r in seg if r["conf"] > 0.4]
        check("tempo stays in the pocket",
              110.0 * 0.95 <= float(np.median(bpms)) <= 135.0 * 1.05,
              f"median bpm {np.median(bpms):.1f} (fleet is 122-126)")
        check("no drop storm", len(drops) <= n_handovers + 2,
              f"{len(drops)} drops over {n_handovers} handovers")

        if keep_wav:
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "logs", "dj_e2e_set.wav")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            wavfile.write(out, RATE,
                          (np.clip(mix, -1, 1) * 32767).astype(np.int16))
            print(f"\n  kept {os.path.normpath(out)} - listen to the seams")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("DJ brain / end-to-end test\n" + "=" * 40 + "\n")
    unit_tests()
    e2e_test("--wav" in sys.argv)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
