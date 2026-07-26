"""FULL-NIGHT SOAK: hours of simulated show through the REAL DJSystem,
engine and library, judged at the scale a live night actually runs.

Everything the per-seam suites verify is re-verified continuously:
level trajectory (no loudness drift as the night wears on), beat lock
in every overlap, repetition spacing, style diversity, clipping, and -
critically - that nothing crashes, stalls, or degrades over hours of
autonomous handovers. This is where slow bugs live.

Usage:
    python tools/tests/_dj_soak_test.py                 # 1.5 simulated hours
    python tools/tests/_dj_soak_test.py --hours 4
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RATE = 44100
BLOCK = 1024
# The SHOW library - measure on the music the DJ actually plays.
MUSIC = "C:/Users/ddehl/Desktop/Devel/music"
import sys as _sys
if "--music" in _sys.argv:
    MUSIC = _sys.argv[_sys.argv.index("--music") + 1]

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def main():
    hours = 1.5
    if "--hours" in sys.argv:
        hours = float(sys.argv[sys.argv.index("--hours") + 1])
    sim_total = hours * 3600.0

    from lib.audio_engine import AudioEngine
    from lib.dj.system import DJSystem
    engine = AudioEngine()
    dj = DJSystem(MUSIC, engine=engine, threaded=False, seed=13)
    assert dj.start(), dj.last_error
    # Soak a PERSONA night: same engine, same judges - the persona layer
    # must hold up to every check the neutral night does.
    if "--persona" in sys.argv:
        from lib.dj.persona import PERSONAS
        name = sys.argv[sys.argv.index("--persona") + 1]
        dj.brain.persona = PERSONAS[name]
        print(f"  persona: {name}")
    gen = engine._mixer()
    next(gen)

    rms_10s = []                 # loudness trajectory
    rms_ctx = []                 # (sim_t, title, deckA, deckB) per window
    grid_lags = []               # (sim_t, ms) during SETTLED overlaps
    dual_run = 0.0               # continuous dual-audible seconds
    plays = []                   # (sim_t, track_id)
    styles = {}
    peak = 0.0
    clipped = 0
    errors = []
    acc = np.zeros(0, dtype=np.float32)
    last_track = None
    wall0 = time.time()
    n_blocks = int(sim_total * RATE) // BLOCK
    for i in range(n_blocks):
        buf = np.frombuffer(gen.send(BLOCK), np.float32).reshape(-1, 2)
        mono = buf.mean(axis=1)
        peak = max(peak, float(np.abs(buf).max()))
        clipped += int((np.abs(buf) > 0.999).sum())
        acc = np.concatenate([acc, mono]) if len(acc) < 10 * RATE else acc
        if len(acc) >= 10 * RATE:
            rms_10s.append(float(np.sqrt((acc[:10 * RATE] ** 2).mean())))
            # Context per window so a quiet-floor failure is diagnosable:
            # a musical dip shows crossing gains on playing decks; a real
            # dropout shows dead decks.
            tel = dj.submix.telemetry or {}
            da = (tel.get("decks") or {}).get("a", {})
            db_ = (tel.get("decks") or {}).get("b", {})
            live = ((da.get('playing') and da.get('gain', 0) > 0.5)
                    or (db_.get('playing') and db_.get('gain', 0) > 0.5))
            rms_ctx.append((
                dj.submix.clock / RATE,
                (dj.current.title if dj.current else "?"),
                f"a:{'P' if da.get('playing') else '-'}"
                f"@{da.get('gain', 0):.2f}",
                f"b:{'P' if db_.get('playing') else '-'}"
                f"@{db_.get('gain', 0):.2f}",
                bool(live)))
            acc = acc[:0]
        if i % 16 == 0:
            try:
                dj.step()
            except Exception as e:
                errors.append(f"{type(e).__name__}: {e}")
                break
        if dj.current is not None and (last_track is None
                                       or dj.current.id != last_track):
            last_track = dj.current.id
            plays.append((dj.submix.clock / RATE, dj.current.id))
            if dj._last_style:
                styles[dj._last_style] = styles.get(dj._last_style, 0) + 1
        if i % 32 == 0:
            tel = dj.submix.telemetry
            if tel:
                da, db_ = tel["decks"]["a"], tel["decks"]["b"]
                both = (da["playing"] and db_["playing"]
                        and da["gain"] > 0.15 and db_["gain"] > 0.15)
                dual_run = dual_run + 32 * BLOCK / RATE if both else 0.0
                # SETTLED only (>2s dual) - the same standard as the seam
                # harness; the snap+PLL convergence window isn't a verdict.
                # SYNCED overlaps only: long_fade/emergency_fade run two
                # free-running grids on purpose (the pair can't be
                # beat-matched), so their "grid delta" is arbitrary - the
                # per-seam quality harness skips them for the same reason.
                if (both and dual_run > 2.0 and tel.get("sync")
                        and not (da.get("braking") or db_.get("braking")
                                 or da.get("loop") or db_.get("loop"))
                        and dj.current is not None):
                    beat = dj.current.period_s
                    # Lock is judged against the PLL's kick-aligned target
                    # (sync offsets the slave's grid by bias_beats).
                    sy = tel.get("sync") or {}
                    bias = float(sy.get("bias_beats") or 0.0)
                    if sy.get("slave") == "a":
                        bias = -bias
                    gd = (dj.submix.decks["a"].beat_phase()
                          - dj.submix.decks["b"].beat_phase()
                          + bias + 0.5) % 1.0 - 0.5
                    grid_lags.append((dj.submix.clock / RATE,
                                      abs(gd) * beat * 1000))
        if i % (n_blocks // 20) == 0 and i:
            print(f"  ... {i / n_blocks * 100:.0f}%  sim {dj.submix.clock / RATE / 60:.0f}min  "
                  f"wall {time.time() - wall0:.0f}s  plays {len(plays)}")

    sim_min = dj.submix.clock / RATE / 60.0
    print(f"\nsoak: {sim_min:.0f} simulated minutes in "
          f"{time.time() - wall0:.0f}s wall | {len(plays)} tracks played\n")

    check("no exceptions over the whole night", not errors,
          errors[0] if errors else "clean")
    check("enough autonomous handovers", len(plays) >= hours * 7,
          f"{len(plays)} plays over {sim_min:.0f} min")
    check("style diversity across the night", len(styles) >= 3,
          f"styles used: {styles}")
    check("no clipping all night", clipped == 0 and peak <= 1.0,
          f"peak {peak:.3f}, clipped samples {clipped}")

    # Loudness: skip the opening ramp, then every 20-min band must sit
    # within +/-2.5 dB of the whole-night median (no quiet-drift).
    r = np.array(rms_10s[3:])
    med = float(np.median(r))
    band = max(len(r) // max(int(sim_min / 20), 1), 4)
    drifts = []
    for b0 in range(0, len(r) - band + 1, band):
        drifts.append(20 * np.log10(
            max(float(np.median(r[b0:b0 + band])), 1e-9) / max(med, 1e-9)))
    check("no loudness drift across the night",
          bool(drifts) and max(abs(d) for d in drifts) <= 2.5,
          f"20-min band deviations: "
          f"{['%+.1f' % d for d in drifts]} dB vs night median")
    # Quiet MUSIC is legitimate (deep breakdowns sit ~-20 dB); a real
    # dropout is near-silence. Gate at -26 dB relative with an absolute
    # floor so true gaps still fail loudly.
    # Quiet MUSIC is legitimate at any depth WHILE A DECK IS LIVE (deep
    # intros on quiet-taste persona nights measured -27 dB relative with a
    # deck at full gain). A real dropout is the relative floor WITHOUT a
    # live deck, or near-absolute silence regardless (a corrupt decode
    # playing zeros must still fail).
    viol = [i for i in range(3, len(rms_10s))
            if rms_10s[i] <= 0.003
            or (rms_10s[i] <= 0.05 * med
                and i < len(rms_ctx) and not rms_ctx[i][4])]
    check("no dead air after the opening", not viol,
          f"min 10s RMS {r.min():.4f} vs median {med:.4f}"
          + (f"; {len(viol)} dead windows" if viol else
             " (quiet floors all had a live deck)"))
    # The 3 quietest windows with deck context, always - a borderline
    # floor should be judged on WHAT it was, not just how quiet.
    if rms_ctx:
        worst = np.argsort(np.array(rms_10s))[:3]
        for i in worst:
            t, title, a, b, _live = rms_ctx[i]
            print(f"    quiet window @{t / 60:5.1f}min rms={rms_10s[i]:.4f} "
                  f"[{a} {b}] {title[:40]}")

    # Beat lock in every overlap, all night (settled samples only).
    lag_vals = [l for _, l in grid_lags]
    if lag_vals:
        check("grid-locked in every overlap",
              float(np.median(lag_vals)) <= 30.0
              and float(np.percentile(lag_vals, 95)) <= 80.0,
              f"{len(lag_vals)} samples: med {np.median(lag_vals):.0f}ms "
              f"p95 {np.percentile(lag_vals, 95):.0f}ms")

    # Repetition spacing in SIMULATED time.
    seen = {}
    worst_gap = None
    for t, tid in plays:
        if tid in seen:
            gap = t - seen[tid]
            worst_gap = gap if worst_gap is None else min(worst_gap, gap)
        seen[tid] = t
    check("no track repeats within a simulated hour",
          worst_gap is None or worst_gap >= 3600.0,
          "no repeats at all" if worst_gap is None
          else f"closest repeat {worst_gap / 60:.0f} min apart")

    print()
    if failures:
        print(f"FAILED: {len(failures)}: {', '.join(failures)}")
        sys.exit(1)
    print("ALL PASS - the night holds together")


if __name__ == "__main__":
    main()
