"""Estimate execution-knob values from RENDERED AUDIO, no ears needed.

The Seam Lab asks a human about the ~9 knobs that are matters of taste.
The other ~30 mostly have measurable consequences - level holes, abrupt
lurches, low-end pile-ups, clipping - which is exactly what this sweeps:

  for each knob: render the SAME handful of seams at several values
  across its range, score every render with objective metrics, and see
  which value the measurements prefer.

Because the same seams are rendered at every value, the pair-to-pair
variance that made verdict statistics hopeless cancels exactly - this is
the A/B design that is impractical for a listener but free for a machine.

A knob whose scores are FLAT across its range is reported as such: the
metrics cannot decide it, so either leave the default or promote it to
the Seam Lab's taste questions (seamprobe.promote).

Usage:
    python tools/dj/dj_knobsweep.py --music D:/Devel/music --knob fade_out_ramp
    python tools/dj/dj_knobsweep.py --music D:/Devel/music --deferred
    ... --apply     # write winners into logs/seam_tuning.json
Results land in logs/knob_sweep.json either way.
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

RATE = 44100
N_VALUES = 5           # points across the knob's range
N_SEAMS = 4            # fixed seams rendered per point
FLAT_DB = 0.8          # score spread below this = "metrics can't decide"
APPLY_MARGIN = 0.5     # winner must beat the default by this to --apply


def metrics(x):
    """Objective badness of one rendered seam, in comparable dB-ish units.

    dead   - deepest 0.5s hole vs the median level (0 = none)
    lurch  - largest second-to-second level step (abrupt = bad)
    lowex  - low-band (<120 Hz) pile-up over its own median (double bass)
    clip   - samples at full scale
    """
    mono = x.mean(axis=1)
    n = len(mono)
    w = RATE // 2
    hops = max((n - w) // (RATE // 4), 1)
    rms = np.array([np.sqrt(np.mean(mono[i * (RATE // 4):
                                         i * (RATE // 4) + w] ** 2)) + 1e-9
                    for i in range(hops)])
    med = np.median(rms)
    dead = max(0.0, 20 * np.log10(med / rms.min()))
    sec = np.array([np.sqrt(np.mean(mono[i * RATE:(i + 1) * RATE] ** 2))
                    + 1e-9 for i in range(max(n // RATE, 1))])
    lurch = float(np.abs(np.diff(20 * np.log10(sec))).max()) \
        if len(sec) > 1 else 0.0
    # low band per second, via rFFT
    lows = []
    for i in range(max(n // RATE, 1)):
        seg = mono[i * RATE:(i + 1) * RATE]
        if len(seg) < RATE // 2:
            continue
        sp = np.abs(np.fft.rfft(seg))
        cut = int(120 * len(seg) / RATE)
        lows.append(np.sqrt(np.mean(sp[:cut] ** 2)) + 1e-9)
    lows = np.array(lows) if lows else np.array([1.0])
    lowex = max(0.0, 20 * np.log10(lows.max() / np.median(lows)) - 6.0)
    clip = int(np.sum(np.abs(x) >= 0.999))
    return {"dead": round(float(dead), 2), "lurch": round(lurch, 2),
            "lowex": round(float(lowex), 2), "clip": clip}


def badness(m):
    """One number, lower is better. Weights are coarse on purpose - this
    ranks values of ONE knob against each other on the SAME seams, so
    only differences matter."""
    return (max(0.0, m["dead"] - 8.0)          # holes deeper than 8 dB
            + 0.8 * max(0.0, m["lurch"] - 6.0)  # steps sharper than 6 dB
            + 0.6 * m["lowex"]
            + 0.1 * min(m["clip"], 50))


def build_seams(db, lib, knob, rng):
    """Fixed (a, b, plan) trios whose style actually reads `knob`."""
    from lib.dj.brain import Brain
    from lib.dj.themes import get_theme
    from tools.dj.planner.seamtune import styles_reading
    brain = Brain(lib, get_theme("groove"), seed=rng.randrange(1 << 30))
    styles = styles_reading(knob)
    out = []
    tries = 0
    while len(out) < N_SEAMS and tries < 300:
        tries += 1
        cur = rng.choice(lib)
        if cur.duration_s < 150:
            continue
        cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
        if cand is None:
            continue
        want = rng.choice(styles)
        plan = brain.plan_transition(cur, cand, meta,
                                     after_s=cur.duration_s * 0.5,
                                     force_style=want, test_gates=True)
        if plan["style"] in styles:
            out.append((cur, cand, plan))
    return out


def sweep_knob(db, lib, knob, rng, log=print):
    from lib.dj.audition import render_seam
    from tools.dj.planner.seamtune import RANGES, apply_plan_knobs
    from lib.dj.brain import TUNE_DEFAULTS
    lo, hi = RANGES[knob]
    values = [round(lo + (hi - lo) * i / (N_VALUES - 1), 4)
              for i in range(N_VALUES)]
    default = TUNE_DEFAULTS[knob]
    seams = build_seams(db, lib, knob, rng)
    if not seams:
        log(f"  {knob}: no seams landed a style that reads it - skipped")
        return None
    log(f"  {knob}: {len(seams)} seams x {N_VALUES} values "
        f"({values[0]}..{values[-1]}, default {default})")
    scores = {}
    for v in values:
        tot = 0.0
        for (a, b, plan) in seams:
            p = dict(plan, tune={knob: v})
            apply_plan_knobs(p)
            audio = render_seam(db, a, b, p)
            tot += badness(metrics(audio))
        scores[v] = round(tot / len(seams), 3)
        log(f"     {knob}={v:<8} badness {scores[v]}")
    best = min(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    # score at (or nearest to) the default, for the apply margin
    near_def = min(scores, key=lambda v: abs(v - default))
    flat = spread < FLAT_DB
    log(f"     -> best {best} (spread {spread:.2f} "
        + ("FLAT - metrics can't decide; taste question or leave default"
           if flat else f"vs default-ish {near_def}={scores[near_def]}") + ")")
    return {"knob": knob, "scores": {str(k): v for k, v in scores.items()},
            "best": best, "default": default, "flat": flat,
            "spread": round(spread, 3),
            "gain_vs_default": round(scores[near_def] - scores[best], 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--music", required=True)
    ap.add_argument("--knob")
    ap.add_argument("--priority", action="store_true",
                    help="sweep the taste-tier knobs with the simulated "
                         "listener (per-deck question metrics)")
    ap.add_argument("--deferred", action="store_true",
                    help="sweep every knob the Seam Lab does not ask about")
    ap.add_argument("--apply", action="store_true",
                    help="write clear winners into the live tuning")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="knob_sweep.json",
                    help="results filename under logs/ (separate files "
                         "let two sweeps run concurrently)")
    args = ap.parse_args()

    from lib.dj.brain import load_library
    from lib.dj.db import LibraryDB
    from tools.dj.planner.seamprobe import PRIORITY, QUESTIONS
    from tools.dj.planner.seamtune import RANGES
    db = LibraryDB(args.music)
    lib = [t for t in load_library(db) if not t.excluded]
    print(f"library: {len(lib)} tracks")

    if args.knob:
        knobs = [args.knob]
    elif args.priority:
        knobs = [k for k in PRIORITY if k in RANGES]
    elif args.deferred:
        knobs = [k for k in RANGES if k in QUESTIONS and k not in PRIORITY]
    else:
        ap.error("--knob NAME, --priority or --deferred")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "logs", args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Resume: knobs already swept (this run or a previous one) are kept
    # and skipped, so a killed sweep costs only the knob in flight.
    results = []
    try:
        with open(out, encoding="utf-8") as f:
            results = [r for r in json.load(f).get("results", [])
                       if r.get("knob") in knobs]
        if results:
            print(f"resuming: {len(results)} knobs already swept")
    except (OSError, ValueError):
        pass
    done = {r["knob"] for r in results}
    rng = random.Random(args.seed)
    t0 = time.time()
    for i, k in enumerate(knobs):
        if k in done:
            continue
        print(f"[{i + 1}/{len(knobs)}] sweeping {k} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)
        fn = (sweep_priority_knob if k in PRIORITY else sweep_knob)
        r = fn(db, lib, k, rng, log=lambda *a: print(*a, flush=True))
        if r:
            results.append(r)
        with open(out, "w", encoding="utf-8") as f:   # after EVERY knob
            json.dump({"t": time.time(), "results": results}, f, indent=2)
    print(f"\nwrote {out}")

    decisive = [r for r in results if not r["flat"]
                and r["gain_vs_default"] >= APPLY_MARGIN]
    flat = [r for r in results if r["flat"]]
    print(f"\n{len(decisive)} knobs where the metrics prefer a non-default "
          f"value; {len(flat)} flat (candidates for taste or leave-alone)")
    for r in decisive:
        print(f"   {r['knob']}: {r['default']} -> {r['best']} "
              f"(saves {r['gain_vs_default']} badness)")
    if args.apply and decisive:
        from lib.dj import tuning
        for r in decisive:
            tuning.set_value(r["knob"], r["best"],
                             why=f"knob sweep: badness "
                                 f"-{r['gain_vs_default']}")
            print(f"   applied {r['knob']} = {r['best']}")


# ===========================================================================
# SIMULATED LISTENER for the taste-tier knobs (--priority).
#
# Each Seam Lab question maps to a measurement on the rendered seam, taken
# PER DECK via post-EQ taps (the trick _dj_spectral_test established):
# "melodies clashed" is both mid-bands hot at once, "barged in" is the
# incoming deck's rise rate, "dragged" is how long both decks sit loud
# together, a "hole" is the mix dropping far below its own median. Too-much
# and too-little both cost, so each knob's badness is U-shaped and the
# sweep can find interior optima. Same seams at every value, so pair
# variance cancels - the machine's A/B.
# ===========================================================================

def render_tapped(db, a, b, plan):
    """audition.render_seam plus full per-deck post-EQ taps.

    Returns (mix, decks, marks): mix (n,2) float32; decks {"a": (n,2),
    "b": (n,2)} aligned sample-for-sample with mix; marks {"blend_s",
    "swap_s"} in render seconds."""
    from lib.audio_engine import AudioEngine
    from lib.dj.brain import Brain
    from lib.dj.features import decode_file_stereo
    from lib.dj.stems import load_stems
    from lib.dj.submix import DJSubmix
    from lib.dj.themes import get_theme

    plan = dict(plan)
    sa = decode_file_stereo(db.abs(a.path))
    sb = decode_file_stereo(db.abs(b.path))
    engine = AudioEngine()
    sub = DJSubmix()
    engine.attach_track("dj", sub)
    pre = max(12.0, plan.get("beats", 32) * 60.0 / max(a.bpm, 60.0) + 6.0)
    cue_a = a.nearest_downbeat(max(0.0, plan["out_s"] - pre))
    sub.post_many([
        {"cmd": "load", "deck": "a", "samples": sa, "grid": a.grid,
         "gain_db": a.gain_db, "cue_s": cue_a},
        {"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01},
        {"cmd": "start", "deck": "a"},
        {"cmd": "load", "deck": "b", "samples": sb, "grid": b.grid,
         "gain_db": b.gain_db, "cue_s": plan["in_s"]},
    ])
    for deck, t, arr in (("a", a, sa), ("b", b, sb)):
        if getattr(t, "has_stems", False):
            st = load_stems(db.music_root, t.id, expected_len=len(arr))
            if st:
                sub.post({"cmd": "attach_stems", "deck": deck,
                          "stems": st})
    gen = engine._mixer()
    next(gen)
    gen.send(256)
    brain = Brain([], get_theme("groove"))
    events, swap_at, blend_at = brain.build_events(
        plan, sub.telemetry, "a", "b", a, b)
    sub.post_many(events)

    # CLOCK-PLACED taps: the mixer skips read() for a deck that is not
    # playing, so appending blocks time-shifts the late-starting deck to
    # the render start (measured: reconstruction residual 1.26). Each
    # block is recorded WITH the submix clock it was read at and scattered
    # into place afterwards.
    taps = {"a": [], "b": []}
    t0 = sub.clock
    for _nm, _d in sub.decks.items():
        def _wrap(orig, nm):
            def f(n):
                blk = orig(n)
                taps[nm].append((sub.clock,
                                 np.asarray(blk, dtype=np.float32)))
                return blk
            return f
        _d.read = _wrap(_d.read, _nm)
    total = pre + (swap_at - blend_at) / RATE + 25.0
    out = [np.frombuffer(gen.send(4410), dtype=np.float32).reshape(-1, 2)
           for _ in range(int(total * RATE) // 4410)]
    mix = np.concatenate(out, axis=0)
    decks = {}
    for nm, blocks in taps.items():
        buf = np.zeros_like(mix)
        for clk, blk in blocks:
            i = clk - t0
            if i < 0 or i >= len(buf):
                continue
            j = min(i + len(blk), len(buf))
            buf[i:j] += blk[:j - i]
        decks[nm] = buf
    return mix, decks, {"blend_s": (blend_at - t0) / RATE,
                        "swap_s": (swap_at - t0) / RATE}


def _bands(x, hop=0.25):
    """Per-hop band RMS in dB: low/mid/high/full. x is (n,2)."""
    from scipy.signal import butter, sosfilt
    mono = x.mean(axis=1).astype(np.float64)
    sigs = {"full": mono}
    for name, kind, freq in (("low", "lowpass", 120.0),
                             ("mid", "bandpass", (250.0, 2000.0)),
                             ("high", "highpass", 4000.0)):
        sos = butter(4, freq, btype=kind, fs=RATE, output="sos")
        sigs[name] = sosfilt(sos, mono)
    w = int(hop * RATE)
    n = max(len(mono) // w, 1)
    out = {}
    for name, sig in sigs.items():
        r = np.array([np.sqrt(np.mean(sig[i * w:(i + 1) * w] ** 2)) + 1e-9
                      for i in range(n)])
        out[name] = 20 * np.log10(r)
    return out


def priority_badness(knob, mix, decks, marks, hop=0.25):
    """Answer the knob's own Seam Lab question from the audio.
    Lower is better; too-much and too-little both cost."""
    A, B, M = _bands(decks["a"], hop), _bands(decks["b"], hop), \
        _bands(mix, hop)
    n = min(len(M["full"]), len(A["full"]), len(B["full"]))
    for d in (A, B, M):
        for k in d:
            d[k] = d[k][:n]
    i_blend = max(int(marks["blend_s"] / hop), 0)
    i_swap = min(int(marks["swap_s"] / hop), n - 1)
    blendw = slice(i_blend, max(i_swap, i_blend + 1))
    both_loud = (A["full"] > A["full"].max() - 12) & \
                (B["full"] > B["full"].max() - 12)
    hole_db = (max(0.0, float(np.median(M["full"])
                              - M["full"][blendw].min()))
               if i_swap > i_blend else 0.0)
    mid_stack = float(np.sum((A["mid"] > A["mid"].max() - 10)
                             & (B["mid"] > B["mid"].max() - 10)) * hop)
    high_stack = float(np.sum((A["high"] > A["high"].max() - 10)
                              & (B["high"] > B["high"].max() - 10)) * hop)

    if knob == "beats_scale":
        drag = max(0.0, float(np.sum(both_loud) * hop) - 20.0)
        flux = np.abs(np.diff(M["full"]))
        rush = (max(0.0, float(flux[max(i_swap - 2, 0):i_swap + 3].max())
                    - 6.0) if i_swap > 2 else 0.0)
        return drag * 0.3 + rush + max(0.0, hole_db - 8) + mid_stack * 0.2
    if knob == "swap_pos":
        a_on = A["full"] > A["full"].max() - 25
        low_gap = float(np.sum((A["low"] < A["low"].max() - 25)
                               & (B["low"] < B["low"].max() - 25)
                               & a_on) * hop)
        low_both = float(np.sum((A["low"] > A["low"].max() - 10)
                                & (B["low"] > B["low"].max() - 10)) * hop)
        return low_gap * 0.5 + low_both * 0.7 + max(0.0, hole_db - 8)
    if knob in ("fade_recede", "fade_b_stage1"):
        mud = float(np.sum(both_loud) * hop)
        rise = np.diff(B["full"])
        barge = max(0.0, float(rise.max()) - 8.0) if len(rise) else 0.0
        return (max(0.0, hole_db - 6) * 1.2
                + max(0.0, mud - 4) * 0.5 + barge)
    if knob == "trim_cap":
        mismatch = (float(np.median(np.abs(A["full"][blendw]
                                           - B["full"][blendw])))
                    if i_swap > i_blend else 0.0)
        return max(0.0, mismatch - 6.0) + max(0.0, hole_db - 8)
    if knob in ("b_mid0", "stage1_gain"):
        share = B["full"][blendw] - A["full"][blendw]
        fight = (float(np.sum(share > 0) * hop)
                 if i_swap > i_blend else 0.0)
        return mid_stack * 0.6 + fight * 0.3 + max(0.0, hole_db - 6)
    if knob == "high_swap_at":
        return high_stack * 0.8 + max(0.0, hole_db - 8)
    if knob == "pre_dip_gain":
        bump = (max(0.0, float(M["full"][blendw].max())
                    - float(np.median(M["full"])) - 3.0)
                if i_swap > i_blend else 0.0)
        return bump + max(0.0, hole_db - 6) * 1.2
    return 0.0


def sweep_priority_knob(db, lib, knob, rng, log=print):
    from lib.dj.brain import TUNE_DEFAULTS
    from tools.dj.planner.seamtune import RANGES, apply_plan_knobs
    lo, hi = RANGES[knob]
    values = [round(lo + (hi - lo) * i / (N_VALUES - 1), 4)
              for i in range(N_VALUES)]
    default = TUNE_DEFAULTS[knob]
    seams = build_seams(db, lib, knob, rng)
    if not seams:
        log(f"  {knob}: no seams landed - skipped")
        return None
    log(f"  {knob}: {len(seams)} seams x {N_VALUES} values "
        f"({values[0]}..{values[-1]}, default {default}) [listener]")
    scores = {}
    for v in values:
        tot = 0.0
        for (a, b, plan) in seams:
            p = dict(plan, tune={knob: v})
            apply_plan_knobs(p)
            mix, decks, marks = render_tapped(db, a, b, p)
            tot += (priority_badness(knob, mix, decks, marks)
                    + badness(metrics(mix)))
        scores[v] = round(tot / len(seams), 3)
        log(f"     {knob}={v:<8} badness {scores[v]}")
    best = min(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    near_def = min(scores, key=lambda v: abs(v - default))
    flat = spread < FLAT_DB
    log(f"     -> best {best} (spread {spread:.2f}"
        + (" FLAT - genuinely a taste call; ask the human"
           if flat else f", default-ish scores {scores[near_def]}") + ")")
    return {"knob": knob, "scores": {str(k): v for k, v in scores.items()},
            "best": best, "default": default, "flat": flat,
            "spread": round(spread, 3), "priority": True,
            "gain_vs_default": round(scores[near_def] - scores[best], 3)}


if __name__ == "__main__":
    main()
