"""HARSH mixing-quality QA: real library, real brain, real submix.

Two parts:
  1. SELECTION AUDIT (fast, no audio) - run the brain's next-track choice
     from every track in the real library; tally styles, stretch ratios,
     pair scores, long_fade fallback rate. Catches "the clever machinery
     never actually engages" failures that per-seam tests can't see.
  2. SEAM RENDERS - brain-planned transitions rendered offline through
     the real submix for several styles on real tracks, measured like a
     hostile listener: audible kick lag between decks, dead air, level
     lurches, double-bass, clipping, cut timing.

Usage:
    python tools/tests/_dj_quality_test.py                # both parts, gate
    python tools/tests/_dj_quality_test.py --audit-only
    python tools/tests/_dj_quality_test.py --wav          # dump seam WAVs to logs/
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from lib.dj import features as F
from lib.dj.brain import Brain, load_library
from lib.dj.db import LibraryDB
from lib.dj.submix import DJSubmix
from lib.dj.themes import get_theme

RATE = 44100
BLOCK = 1024
# The SHOW library is the default - quality must be measured on the music
# the DJ actually plays, not the clean dev fleet. Override: --music <dir>.
MUSIC = "C:/Users/ddehl/Desktop/Devel/music"
if "--music" in sys.argv:
    MUSIC = sys.argv[sys.argv.index("--music") + 1]

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# ==========================================================================
# Part 1: selection audit
# ==========================================================================

def selection_audit(library, theme):
    print("\n=== selection audit: brain choice from every library track ===")
    styles, rates, scores = {}, [], []
    no_pick = 0
    fades = 0
    for i, cur in enumerate(library):
        brain = Brain(library, theme, seed=i)     # fresh recency each time
        brain.note_played(cur)
        cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
        if cand is None:
            no_pick += 1
            continue
        plan = brain.plan_transition(cur, cand, meta,
                                     after_s=cur.duration_s * 0.45)
        styles[plan["style"]] = styles.get(plan["style"], 0) + 1
        rates.append(abs(plan["rate"] - 1.0))
        scores.append(plan["pair_score"])
        if plan["style"] == "long_fade":
            fades += 1
    n = len(library) - no_pick
    print(f"  picks: {n}/{len(library)}  styles: "
          + " ".join(f"{k}={v}" for k, v in sorted(styles.items())))
    print(f"  |rate-1|: median {np.median(rates)*100:.1f}% "
          f"p95 {np.percentile(rates, 95)*100:.1f}% max {max(rates)*100:.1f}%")
    print(f"  pair_score: median {np.median(scores):.3f} "
          f"min {min(scores):.3f}")
    check("brain finds a next track", no_pick <= len(library) * 0.05,
          f"{no_pick} tracks had no compatible successor")
    # Stretch discipline: the selection wall sits at 5.5%; beyond it only
    # dead-end rescues pass (soft x0.05), clamped at the physical 8%.
    check("stretch discipline",
          np.median(rates) <= 0.02 and np.percentile(rates, 95) <= 0.06
          and max(rates) <= 0.081,
          f"median {np.median(rates)*100:.1f}% p95 "
          f"{np.percentile(rates, 95)*100:.1f}% max {max(rates)*100:.1f}% "
          f"(wall 5.5%, rescue clamp 8%)")
    # long_fade share floor is set by the LIBRARY: any seam touching a
    # low-confidence grid MUST fade (blending unmixable material is worse).
    # This audit starts from EVERY track, so the floor is the low-conf
    # share itself; beyond floor+margin means the machinery isn't engaging.
    lc_share = sum(1 for t in library if t.bpm_conf < 0.5) / len(library)
    fade_bar = max(0.30, lc_share + 0.28)
    check("long_fade is the exception, not the rule",
          fades <= n * fade_bar,
          f"{fades}/{n} seams fall back to long_fade "
          f"({fades/max(n,1)*100:.0f}%; bar {fade_bar*100:.0f}% = "
          f"low-conf share {lc_share*100:.0f}% + 28)")
    check("style variety actually used", len(styles) >= 4,
          f"{len(styles)} distinct styles chosen: {sorted(styles)}")
    check("pair scores not collapsed", np.median(scores) > 0.05,
          f"median pair score {np.median(scores):.3f}")
    return styles


# ==========================================================================
# Part 2: seam renders
# ==========================================================================

def force_style(theme, style):
    t = get_theme(theme.name)
    t.style_weights = {k: (1.0 if k == style else 0.0)
                       for k in t.style_weights}
    return t


def render_seam(library, cur, style, wav=False):
    """Arm one brain-planned transition exactly like DJSystem does and
    render it offline. Returns (metrics dict | None if style not legal)."""
    theme = force_style(get_theme("groove"), style)
    brain = Brain(library, theme, seed=7)
    brain.note_played(cur)
    cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
    if cand is None:
        return None
    plan = brain.plan_transition(cur, cand, meta,
                                 after_s=cur.duration_s * 0.45)
    if plan["style"] != style:
        return None                       # gates said no (no drop/loop/...)

    a = F.decode_file_stereo(os.path.join(MUSIC, cur.path))
    b = F.decode_file_stereo(os.path.join(MUSIC, cand.path))

    from lib.audio_engine import AudioEngine
    engine = AudioEngine()
    sub = DJSubmix()
    engine.attach_track("dj", sub)
    # The render must start BEFORE the blend does: build_events clamps the
    # blend start to 'now', so a fixed 20s pre-roll silently truncated
    # every 64-beat blend to its last ~20s - the gate measured (and
    # blessed) blends the live system never actually played that short.
    pre_roll = max(20.0, plan["beats"] * cur.period_s + 8.0)
    cue_a = max(cur.nearest_downbeat(plan["out_s"] - pre_roll), 0.0)
    sub.post({"cmd": "load", "deck": "a", "samples": a, "grid": cur.grid,
              "gain_db": cur.gain_db, "cue_s": cue_a})
    sub.post({"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01})
    sub.post({"cmd": "start", "deck": "a"})

    gen = engine._mixer()
    next(gen)
    rendered = []
    for _ in range(int(2.0 * RATE) // BLOCK):     # telemetry warm-up
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))

    sub.post({"cmd": "load", "deck": "b", "samples": b, "grid": cand.grid,
              "gain_db": cand.gain_db, "cue_s": plan["in_s"]})
    for _ in range(4):
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))
    events, swap_at, blend_at = brain.build_events(
        plan, sub.telemetry, "a", "b", cur, cand)
    sub.post_many(events)

    from lib.dj.deck import ENV_FPS
    beat = cur.period_s
    end_clock = swap_at + int(10.0 * RATE)
    lags, grid_lags, dual = [], [], 0.0
    tel_log = []
    # PER-DECK MID-BAND TAP (250-2500 Hz - where melodies live): wrap each
    # deck's read so we can verify ONE MELODY AT A TIME in the actual
    # rendered audio, not just in the scheduled events.
    from scipy.signal import butter as _butter, sosfilt as _sf, sosfilt_zi
    _mid_sos = _butter(2, [250.0, 2500.0], btype="band", fs=RATE,
                       output="sos")
    mid_tap = {"a": [], "b": []}
    _zi = {}
    for _nm, _d in sub.decks.items():
        _zi[_nm] = np.stack([sosfilt_zi(_mid_sos) * 0.0 for _ in range(2)])

        def _wrap(orig, nm):
            def f(n):
                blk = orig(n)
                filt = np.empty_like(blk)
                for c in range(2):
                    filt[:, c], _zi[nm][c] = _sf(_mid_sos, blk[:, c],
                                                 zi=_zi[nm][c])
                mid_tap[nm].append((sub.clock,
                                    float(np.sqrt((filt ** 2).mean()))))
                return blk
            return f
        _d.read = _wrap(_d.read, _nm)
    i = len(rendered)
    while sub.clock < end_clock:
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))
        i += 1
        tel = sub.telemetry
        if not tel:
            continue
        da, db_ = tel["decks"]["a"], tel["decks"]["b"]
        tel_log.append((tel["clock"], da["gain"], db_["gain"],
                        da["rate"], db_["rate"], da["playing"],
                        db_["playing"]))
        both = (da["playing"] and db_["playing"]
                and da["gain"] > 0.15 and db_["gain"] > 0.15)
        if both:
            dual += BLOCK / RATE
            # GROUND TRUTH: grid-phase delta between the decks. The env
            # xcorr below is what a listener's ear might latch onto, but
            # on organic material it often measures rhythm-PATTERN offset
            # (shakers vs congas), not flam - grids are the arbiter.
            braking = da.get("braking") or db_.get("braking")
            looping = bool(da.get("loop")) or bool(db_.get("loop"))
            if i % 8 == 0 and not braking and not looping:
                # Sync intentionally offsets the slave's GRID by the kick-
                # alignment bias (kicks in register, not grids) - measure
                # lock against that target, not raw grid coincidence.
                sy = tel.get("sync") or {}
                bias = float(sy.get("bias_beats") or 0.0)
                if sy.get("slave") == "a":
                    bias = -bias
                gd = (sub.decks["a"].beat_phase()
                      - sub.decks["b"].beat_phase() + bias + 0.5) % 1.0 - 0.5
                grid_lags.append((dual, abs(gd) * beat * 1000))
            # A looping/stuttering deck breaks envelope correlation - skip.
            # And require a CONFIDENT beat pattern (same 1.8x-rms gate the
            # sync itself uses): pre-drop/build run-ins have no transients,
            # so their xcorr argmax is noise, not an audible flam.
            loopy = bool(da.get("loop")) or bool(db_.get("loop"))
            if i % 32 == 0 and not loopy:
                em = np.maximum(np.diff(
                    sub.decks["a"].out_env[-300:].astype(np.float64)), 0.0)
                es = np.maximum(np.diff(
                    sub.decks["b"].out_env[-300:].astype(np.float64)), 0.0)
                if em.max() > 1e-4 and es.max() > 1e-4:
                    em -= em.mean()
                    es -= es.mean()
                    xc = np.correlate(em, es, "full")
                    mid = len(es) - 1
                    half = max(int(0.5 * beat * ENV_FPS), 2)
                    seg = xc[mid - half:mid + half + 1]
                    k = int(np.argmax(seg))
                    peak = float(seg[k])
                    rms_c = float(np.sqrt(np.mean(seg ** 2))) + 1e-12
                    if peak > 0 and peak >= 1.8 * rms_c:
                        lags.append((dual, abs(k - half) / ENV_FPS * 1000))
    mix = np.concatenate(rendered, axis=0)
    mono = mix.mean(axis=1).astype(np.float64)

    # Settled lag: after 2 s of dual-audible the PLL has measured at least
    # twice - launch convergence is reported separately, judged gentler.
    early = [l for d, l in lags if d <= 2.0]
    settled = [l for d, l in lags if d > 2.0]
    m = {"pair": f"{cur.title[:24]} -> {cand.title[:24]}",
         "style": style, "rate": plan["rate"],
         "dual_s": dual, "n_lags": len(settled),
         "lag_early": float(np.median(early)) if early else None,
         "lag_med": float(np.median(settled)) if settled else None,
         "lag_max": float(np.max(settled)) if settled else None,
         "raw_lags": lags, "grid_lags": grid_lags,
         "peak": float(np.abs(mix).max()),
         "clipped": int((np.abs(mix) > 0.999).sum())}

    # Dead air / lurch on 0.5 s RMS windows (skip first 1 s).
    w = RATE // 2
    rms = np.sqrt(np.mean(mono[:len(mono)//w*w].reshape(-1, w)**2, axis=1))
    act = rms[2:]
    m["rms_min_ratio"] = float(act.min() / max(np.median(act), 1e-9))
    db_steps = 20 * np.abs(np.diff(np.log10(np.maximum(act, 1e-6))))
    # Lurches judged against the PAIR'S OWN solo dynamics: a drop is a
    # legitimate 7 dB step - the transition just must not lurch harder
    # than the music does on its own.
    start_clock = sub.clock - len(mono)
    bw0 = max(int((blend_at - start_clock) / RATE / 0.5) - 2, 0)
    # The blend's MUSICAL end: past it the incoming track carries the mix
    # alone and its own arrangement moves (first bass entrance!) are its
    # music, not our transition. 6 beats before the stop event (was 4):
    # with the swap moved mid-blend (2026-07-12) A's exit fade completes
    # ~end, i.e. ~4 beats before stop - and a VERIFIED B-solo 6.0 dB
    # arrangement step (Blinding Lights remix, B rendered alone) landed
    # half a window inside the old boundary and read as a 6.6 dB
    # transition lurch while every step we actually schedule measured
    # <= 4.4 dB. A is below -18 dB for the extra second this excludes.
    b_end = swap_at - int(6 * cur.period_s * RATE)
    bw1 = max(int((b_end - start_clock) / RATE / 0.5) - 2, bw0 + 1)
    inside = db_steps[bw0:bw1]
    outside = np.concatenate([db_steps[:bw0], db_steps[bw1:]])
    m["lurch_db"] = float(inside.max()) if len(inside) else 0.0
    m["lurch_solo_db"] = float(outside.max()) if len(outside) else 0.0
    if "--diag" in sys.argv:
        top = np.argsort(db_steps)[-3:][::-1]
        b0s = (blend_at - start_clock) / RATE
        m["worst_steps"] = [
            (round((k + 3) * 0.5 - b0s, 1), round(float(db_steps[k]), 1))
            for k in top]
        m["events"] = [(round((e["at"] - blend_at) / RATE, 1), e["cmd"],
                        e.get("deck", e.get("slave", "")))
                       for e in sorted(events, key=lambda e: e["at"])]
        m["swap_rel_s"] = round((swap_at - blend_at) / RATE, 1)

    # Double-bass: low band (<130 Hz) median level during dual vs solo.
    from scipy.signal import butter, sosfilt
    sos = butter(4, 130.0, fs=RATE, output="sos")
    low = sosfilt(sos, mono)
    lo_rms = np.sqrt(np.mean(low[:len(low)//w*w].reshape(-1, w)**2, axis=1))
    blend_w0 = max(int((blend_at - (sub.clock - len(mono))) / RATE / 0.5), 0)
    swap_w = int((swap_at - (sub.clock - len(mono))) / RATE / 0.5)
    pre = lo_rms[2:max(blend_w0, 3)]
    mid_seg = lo_rms[blend_w0:max(swap_w, blend_w0 + 1)]
    post = lo_rms[swap_w:]
    if len(pre) and len(mid_seg) and len(post):
        # Judged against the pair's own SOLO low-band PEAKS (p95 of the
        # windows where one deck carries the mix alone), not the median:
        # a bass-dynamic track swings its OWN low band 7+ dB (measured
        # 2026-07-13: 'Axe Nord Sud' solo p95/median = +7.4 dB, rendered
        # alone at the same cue/rate), and the old median reference read
        # that crest as double bass. Same lesson as the lurch gate: the
        # transition must not out-peak the music's own peaks - but the
        # music's peaks are its own business.
        solo = max(np.percentile(pre, 95), np.percentile(post, 95))
        m["bass_bump_db"] = float(20 * np.log10(
            max(mid_seg.max(), 1e-9) / max(solo, 1e-9)))
    else:
        m["bass_bump_db"] = 0.0

    # ONE MELODY AT A TIME: both decks' mid-bands (250-2500 Hz) at
    # substantial level together only during the swap handover.
    def _bins(tap):
        out = {}
        for c, v in tap:
            out.setdefault(c // (RATE // 4), []).append(v)
        return {k: max(v) for k, v in out.items()}
    ba, bb = _bins(mid_tap["a"]), _bins(mid_tap["b"])
    pa = max(ba.values(), default=1e-9)
    pb = max(bb.values(), default=1e-9)
    m["mid_overlap_s"] = round(sum(
        0.25 for k in set(ba) & set(bb)
        if ba[k] > 0.4 * pa and bb[k] > 0.4 * pb), 2)

    if wav:
        from scipy.io import wavfile
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                         "logs", f"dj_qa_{style}.wav")
        wavfile.write(p, RATE,
                      (np.clip(mix, -1, 1) * 32767).astype(np.int16))
        m["wav"] = os.path.normpath(p)
    return m


def seam_qa(library, wav=False):
    print("\n=== seam renders: brain-planned transitions, real tracks ===")
    # A busy, confident, drop-having track makes every style legal.
    cands = sorted([t for t in library
                    if t.bpm_conf > 0.6 and t.downbeat_conf > 0.3
                    and t.duration_s > 240], key=lambda t: -t.rhythm_density)
    styles = ["bass_swap", "long_blend", "cut_at_drop", "loop_build",
              "double_drop", "loop_roll_exit", "bassline_layer",
              "filter_sweep", "echo_out", "long_fade"]
    # long_fade engages on LOW-confidence grids - use that pool for it.
    fade_cands = sorted([t for t in library
                         if t.bpm_conf < 0.45 and t.duration_s > 240],
                        key=lambda t: -t.rhythm_density)
    got = {}
    for style in styles:
        m = None
        pool = fade_cands if style == "long_fade" else cands
        for cur in pool[:12]:
            try:
                m = render_seam(library, cur, style, wav=wav)
            except Exception as e:
                print(f"  [FAIL] {style} render crashed on {cur.title[:30]}: "
                      f"{type(e).__name__}: {e}")
                failures.append(f"{style} crash")
                m = None
                break
            if m:
                break
        if m is None:
            print(f"  [warn] {style}: no legal pair found in top candidates")
            continue
        got[style] = m
        lag = (f"lag settled med {m['lag_med']:.0f}ms max {m['lag_max']:.0f}ms"
               if m["lag_med"] is not None else "no settled dual window")
        if m["lag_early"] is not None:
            lag += f" (launch {m['lag_early']:.0f}ms)"
        print(f"\n  {style}: {m['pair']}  rate={m['rate']:.3f}")
        if "worst_steps" in m:
            print(f"    worst steps (s after blend start, dB): "
                  f"{m['worst_steps']}  swap at +{m['swap_rel_s']}s")
            print(f"    events: {m['events']}")
        print(f"    dual {m['dual_s']:.1f}s | {lag} | peak {m['peak']:.2f} "
              f"clip {m['clipped']} | rms_min {m['rms_min_ratio']:.2f} "
              f"lurch {m['lurch_db']:.1f}dB (solo {m['lurch_solo_db']:.1f}) "
              f"| bass bump {m['bass_bump_db']:+.1f}dB")
        check(f"{style}: no dead air", m["rms_min_ratio"] > 0.15,
              f"min/median RMS {m['rms_min_ratio']:.2f}")
        if style != "double_drop":     # its giant synchronized slam IS the
            check(f"{style}: no unmusical lurch",      # style's contract
                  m["lurch_db"] <= max(m["lurch_solo_db"], 4.0) + 2.5,
                  f"blend step {m['lurch_db']:.1f} dB vs solo "
                  f"{m['lurch_solo_db']:.1f} dB")
        check(f"{style}: no clipping", m["clipped"] == 0
              and m["peak"] <= 1.0, f"peak {m['peak']:.3f} "
              f"clipped {m['clipped']}")
        check(f"{style}: no double bass", m["bass_bump_db"] < 3.5,
              f"blend low-band bump {m['bass_bump_db']:+.1f} dB")
        if style == "long_fade":
            # The dipped handoff: the two songs may BOTH be loud for only
            # a moment - a 12s full-range wash on an unmixable pair was
            # exactly what 'terribly mixed' sounded like.
            check("long_fade: overlap is a dip, not a wash",
                  m["mid_overlap_s"] <= 3.5,
                  f"both mid-bands hot {m['mid_overlap_s']:.1f}s "
                  f"(dip budget 3.5)")
        elif style != "double_drop":
            # double_drop stacks full-range on purpose. Everything else:
            # one melody at a time.
            check(f"{style}: one melody at a time",
                  m["mid_overlap_s"] <= 4.0,
                  f"both mid-bands hot {m['mid_overlap_s']:.1f}s "
                  f"(handover budget 4.0)")
        # Sync verdict on GRID delta (settled: dual > 2 s); the env-xcorr
        # lag stays in the report as the ear's-eye view but on organic
        # percussion it conflates pattern offset with flam.
        gl = [l for d, l in m["grid_lags"] if d > 2.0]
        if gl and style != "long_fade":     # fade decks are unsynced by design
            # Short-dual accent styles (a few bars, PLL barely settles;
            # conf-gated >=0.7 both sides + stretch-walled live) get a
            # wider bar than the long blends the night is built on.
            med_bar = 35.0 if style in ("double_drop", "echo_out",
                                        "cut_at_drop") else 25.0
            check(f"{style}: decks grid-locked",
                  float(np.median(gl)) <= med_bar
                  and float(np.percentile(gl, 95)) <= 60.0,
                  f"grid delta med {np.median(gl):.0f}ms "
                  f"p95 {np.percentile(gl, 95):.0f}ms "
                  f"(harsh: {med_bar:.0f}/60)")
        min_dual = {"bass_swap": 4.0, "long_blend": 4.0, "double_drop": 4.0,
                    "cut_at_drop": 3.0, "bassline_layer": 4.0}.get(style)
        if min_dual and m["dual_s"] < min_dual:
            check(f"{style}: decks actually overlap", False,
                  f"dual-audible only {m['dual_s']:.1f}s "
                  f"(need {min_dual:.0f})")
    check("style coverage rendered", len(got) >= 4,
          f"rendered {sorted(got)} of {styles}")
    return got


def main():
    db = LibraryDB(MUSIC)
    library = load_library(db)
    print(f"library: {len(library)} tracks")
    theme = get_theme("groove")
    selection_audit(library, theme)
    if "--audit-only" not in sys.argv:
        seam_qa(library, wav="--wav" in sys.argv)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s):")
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
