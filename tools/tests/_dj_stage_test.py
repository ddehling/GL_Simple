"""Gate: the stem stage holds several lanes on one clock.

Three stems of three tracks (a drums lane, then a bass lane, then a melodic
lane) are brought in one after another on the real DJSubmix through the
AudioEngine's mixer generator - the same path the planner's Stage tab and a
night would run - held together for bars, then the master lane leaves and
the clock hands over. Measured from the submix's own telemetry and audio:

  1. LOCK: every slave lane's beat-phase error against the master (bias
     removed) once it has been in for two bars - median under 0.05 beat,
     95th percentile under 0.12 (a tenth of a beat at 125 bpm is 48 ms;
     the flam the ear forgives is ~30 ms, the PLL's own bar).
  2. HANDOVER: after the master leaves, the new master's slaves lock again
     within four bars.
  3. LEVEL: no bar of the rendered stage more than 6 dB under the bar before
     it while two or more lanes are live (no holes on the way in), peak
     under 0.99 (no clipping).
  4. PLUMBING: the harmonic guard refuses a clashing melodic lane unless
     allowed; a lane's loop wraps on its bar (the deck's loop is exactly
     n bars of the track's own grid).

    python tools/tests/_dj_stage_test.py --music D:/Devel/music [--wav out.wav] [--ids 901 696 581]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

RATE = 44100
FAILS = []


def check(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def main():
    from lib.audio_engine import AudioEngine
    from lib.dj.db import LibraryDB
    from lib.dj import brain as B
    from lib.dj.stage import Stage

    music = sys.argv[sys.argv.index("--music") + 1] if "--music" in sys.argv else "D:/Devel/music"
    ids = [int(x) for x in sys.argv[sys.argv.index("--ids") + 1:sys.argv.index("--ids") + 4]] if "--ids" in sys.argv else [901, 696, 581]
    db = LibraryDB(music)
    lib = {t.id: t for t in B.load_library(db)}
    ta, tb, tc = (lib[i] for i in ids)
    print(f"== stage: {ta.title[:24]} drums | {tb.title[:24]} bass | {tc.title[:24]} other")

    engine = AudioEngine()
    stage = Stage(db, music, n_lanes=4)
    engine.attach_track("stage", stage.submix)
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    audio = []
    errs = {}                              # slave deck -> [(bar_index, |err| beats)]

    def run(seconds):
        """Render `seconds` through the mixer, sampling the lock every block."""
        n = int(seconds * RATE) // 4410
        for _ in range(n):
            audio.append(np.frombuffer(gen.send(4410), dtype=np.float32).reshape(-1, 2))
            stage.refresh()
            tel = stage.submix.telemetry or {}
            sync = tel.get("sync") or {}
            views = sync.get("slaves") or ({sync["slave"]: sync} if sync else {})
            decks = tel.get("decks") or {}
            for sl, v in views.items():
                m = decks.get(v["master"]) or {}
                s = decks.get(sl) or {}
                if not (m.get("playing") and s.get("playing")):
                    continue
                e = (float(s["beat_phase"]) - float(m["beat_phase"]) - float(v.get("bias_beats") or 0.0) + 0.5) % 1.0 - 0.5
                errs.setdefault(sl, []).append((len(audio), abs(e)))

    # material (decoded once, off any audio thread here)
    stage.load(0, ta)
    stage.load(1, tb)
    stage.load(2, tc)
    gen.send(4410)                         # let the loads apply

    def groove(t):
        secs = t.sections or []
        for i, s in enumerate(secs):
            if s.get("kind") == "groove" and s.get("bass_share", 0.3) >= 0.28:
                return i
        return 0

    bar_a = 4 * ta.period_s
    # 1) drums of A: the clock
    at0 = stage.arm(0, "drums", groove(ta), loop_bars=8)
    check(at0 is not None, "drums lane armed (master)")
    run(4 * bar_a)
    st = stage.status()
    check(st["master"] == "lane0" and st["lanes"][0]["playing"], f"lane0 is the master and playing (master {st['master_bpm']:.1f} bpm)")
    # 2) bass of B joins
    at1 = stage.arm(1, "bass", groove(tb), loop_bars=8)
    check(at1 is not None and at1 > at0, "bass lane armed on a later bar")
    run(8 * bar_a)
    # 3) the melodic stem of C joins (through the harmonic guard)
    r = stage.lanes[2]
    at2 = stage.arm(2, "other", groove(tc), loop_bars=8)
    if at2 is None and r.clash:
        print(f"  --   {tc.title[:24]} other clashes with the master key ({tc.camelot} vs {stage.master_camelot}); allowed for the test")
        at2 = stage.arm(2, "other", groove(tc), loop_bars=8, allow_clash=True)
    check(at2 is not None, f"melodic lane armed (shift {stage.lanes[2].key_shift:+d} st, compat {stage.lanes[2].compat})")
    run(8 * bar_a)
    n_before_release = len(audio)
    # 4) the master leaves: the clock hands over
    stage.release(0)
    run(6 * bar_a)
    st = stage.status()
    check(st["master"] == "lane1", f"the clock handed over to lane1 (master now {st['master']})")
    check(not st["lanes"][0]["playing"], "lane0 stopped after its exit")
    # 5) loop plumbing
    l1 = stage.lanes[1]
    bar_b = stage.bar_len_s(tb, l1.loop[0])
    check(abs((l1.loop[1] - l1.loop[0]) / bar_b - 8) < 1e-3, f"lane1 loop is exactly 8 bars of its grid ({(l1.loop[1] - l1.loop[0]):.3f}s)")
    check(abs(l1.rate - stage.master_bpm / tb.bpm) < 1e-6, f"lane1 rate {l1.rate:.4f} = master/track tempo")

    # lock figures, two bars after each arm (blocks of 0.1 s; a bar of A in blocks)
    blk_per_bar = bar_a / 0.1
    for sl, rows in errs.items():
        if not rows:
            continue
        first = rows[0][0]
        settled = [e for (k, e) in rows if k >= first + 2 * blk_per_bar]
        if len(settled) < 10:
            continue
        med, p95 = float(np.median(settled)), float(np.percentile(settled, 95))
        check(med < 0.05 and p95 < 0.12, f"{sl} lock: median {med:.3f} beat, p95 {p95:.3f} (n {len(settled)})")
    # after the handover: lane2 follows lane1 - its error series continues under the new master
    post = [e for (k, e) in errs.get("lane2", []) if k >= n_before_release + 4 * blk_per_bar]
    if post:
        check(float(np.median(post)) < 0.06, f"lane2 re-locked to the new master: median {np.median(post):.3f} beat (n {len(post)})")
    else:
        check(False, "lane2 produced no lock readings after the handover")

    # level and clipping
    x = np.concatenate(audio, axis=0)
    mono = x.mean(axis=1)
    peak = float(np.abs(x).max())
    check(peak < 0.99, f"peak {peak:.3f}")
    nb = int(bar_a * RATE)
    bars = [20 * np.log10(np.sqrt(np.mean(mono[i:i + nb] ** 2)) + 1e-9) for i in range(0, len(mono) - nb, nb)]
    two_live_from = int(((at1 or 0) - 0) / RATE / bar_a) + 1
    steps = [bars[i] - bars[i - 1] for i in range(two_live_from + 1, min(len(bars), n_before_release // nb))]
    worst = min(steps) if steps else 0.0
    check(worst > -6.0, f"no hole while lanes hold: worst bar-to-bar step {worst:+.1f} dB")
    if "--wav" in sys.argv:
        import soundfile as sf
        out = sys.argv[sys.argv.index("--wav") + 1]
        sf.write(out, x, RATE, subtype="PCM_16")
        print(f"  --   wrote {out} ({len(x) / RATE:.0f} s)")
    for line in stage.log:
        print("      " + line)
    db.close()
    print("\nALL OK" if not FAILS else f"\n{len(FAILS)} FAIL: " + "; ".join(FAILS))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
