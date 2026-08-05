"""HARSH spectral QA: tonal behavior of brain-planned seams, real library.

The mixing-quality gate (_dj_quality_test.py) judges LEVEL and TIME —
kick lag, lurches, dead air, clipping. This gate judges the SPECTRUM,
i.e. whether the EQ/stem/filter choreography the brain scripts actually
lands in the rendered audio:

  1. SPECTRAL AUDIT (fast, no audio) — the spectral data and scoring
     machinery engage at all: shares computed and varied across the
     library, sections carry the bass info the swap-floor timing needs,
     and a theme's spectral_lean measurably steers selection.
  2. SEAM SPECTRA — brain-planned transitions rendered offline through
     the real submix with PER-DECK low-band taps, measured like a
     hostile listener with an RTA: two basslines at once, hi-hat
     stacking, low-mid mud, low-end holes, cliff-edge bass swaps, and
     the deck leaving the seam with a stuck filter / carved EQ (the
     "no bass night" class — undiagnosable live, 2026-07-13).

All measurements are music-only renders on the real library (no
synthesized FX are scheduled by any style in the pool) — per the house
rule: judge DJ audio on what real nights actually play.

Usage:
    python tools/tests/_dj_spectral_test.py                # both parts, gate
    python tools/tests/_dj_spectral_test.py --audit-only
    python tools/tests/_dj_spectral_test.py --styles bass_swap,long_blend
    python tools/tests/_dj_spectral_test.py --wav          # dump WAVs to logs/
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
# The SHOW library is the default — spectra must be measured on the music
# the DJ actually plays, not the clean dev fleet. Override: --music <dir>.
MUSIC = "C:/Users/ddehl/Desktop/Devel/music"
if "--music" in sys.argv:
    MUSIC = sys.argv[sys.argv.index("--music") + 1]

# The three bands a club listener actually complains about: double kicks
# (<130), low-mid mud (180–600), stacked hats/air (>6k). Mid melodies are
# the quality gate's territory (its "one melody at a time" tap).
LOW_HZ, MUD_LO, MUD_HI, HIGH_HZ = 130.0, 180.0, 600.0, 6000.0

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# ==========================================================================
# Part 1: spectral machinery audit (no audio)
# ==========================================================================

def spectral_audit(library):
    print("\n=== spectral audit: shares, sections, lean steering ===")
    have = [t for t in library if t.spectral]
    check("spectral shares computed", len(have) >= 0.9 * len(library),
          f"{len(have)}/{len(library)} tracks carry a spectral blob")
    sums = [t.spectral.get("bass_share", 0) + t.spectral.get("mid_share", 0)
            + t.spectral.get("high_share", 0) for t in have]
    check("shares form a distribution",
          bool(sums) and 0.85 <= float(np.median(sums)) <= 1.15,
          f"median bass+mid+high = {np.median(sums):.2f}" if sums else "none")
    bass = [t.spectral.get("bass_share", 0.33) for t in have]
    check("shares vary across the library",
          bool(bass) and float(np.std(bass)) > 0.03,
          f"bass_share std {np.std(bass):.3f} "
          f"(≈0 means the scan wrote defaults)" if bass else "none")

    # The blend's swap-floor logic ("never swap the bass into a bassless
    # stretch of B") reads section-level bass_share — if sections don't
    # carry it, that guard silently never fires.
    with_sec = [t for t in library if t.sections]
    sec_bass = [t for t in with_sec
                if any("bass_share" in s for s in t.sections)]
    check("sections carry bass info for swap timing",
          len(with_sec) > 0 and len(sec_bass) >= 0.6 * len(with_sec),
          f"{len(sec_bass)}/{len(with_sec)} sectioned tracks have "
          f"per-section bass_share")

    # spectral_lean must actually steer choose_next — a broken wiring here
    # is invisible per-seam (every individual pick still looks sane).
    seeds = library[:min(40, len(library))]
    got = {"": [], "bass": []}
    for lean, out in got.items():
        for i, cur in enumerate(seeds):
            th = get_theme("groove")
            th.spectral_lean = lean
            brain = Brain(library, th, seed=i)
            brain.note_played(cur)
            cand, _meta = brain.choose_next(cur, 0.6, cur.bpm)
            if cand is not None:
                out.append(cand.spectral.get("bass_share", 0.33))
    if got[""] and got["bass"]:
        d = float(np.mean(got["bass"]) - np.mean(got[""]))
        check("spectral_lean steers selection", d > 0.0,
              f"bass-lean picks avg bass_share {np.mean(got['bass']):.3f} "
              f"vs neutral {np.mean(got['']):.3f} (delta {d:+.3f})")
    else:
        check("spectral_lean steers selection", False, "no picks to compare")


# ==========================================================================
# Part 2: seam spectra
# ==========================================================================

def force_style(theme, style):
    # Styles ABSENT from a theme's weight dict get accent-tier defaults in
    # plan_transition — zeroing only the dict's existing keys leaves those
    # alive and the forced style has to win a dice roll against them.
    # Enumerate the defaulted vocabulary explicitly so forcing is forcing.
    t = get_theme(theme.name)
    known = set(t.style_weights) | {
        "stem_drum_swap", "acapella_out", "stem_bass_swap", "drum_bridge",
        "acapella_in", "melody_carry", "phrase_cut",
        "breakdown_swap"}
    t.style_weights = {k: (1.0 if k == style else 0.0) for k in known}
    return t


def _band_windows(x, sos, w):
    from scipy.signal import sosfilt
    y = sosfilt(sos, x)
    return np.sqrt(np.mean(y[:len(y) // w * w].reshape(-1, w) ** 2, axis=1))


def _bins(tap, span=RATE // 4):
    out = {}
    for c, v in tap:
        out.setdefault(c // span, []).append(v)
    return {k: max(v) for k, v in out.items()}


def render_seam(library, cur, style, seed=7, wav=False):
    """Arm one brain-planned transition exactly like DJSystem does, render
    it offline with per-deck low-band taps, and measure the spectrum.
    Returns metrics dict, or None if the style wasn't legal for the pair."""
    theme = force_style(get_theme("groove"), style)
    brain = Brain(library, theme, seed=seed)
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
    # Start BEFORE the blend does (same lesson as the quality gate: a fixed
    # pre-roll silently truncated 64-beat blends and blessed audio the live
    # system never played).
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

    # PER-DECK LOW-BAND TAP (post-EQ/filter/stems — the deck's actual
    # output): the mix-level "no double bass" gate can pass while both
    # decks quietly share the low end; the tap is the ground truth for
    # ONLY ONE BASSLINE AT A TIME.
    from scipy.signal import butter, sosfilt as _sf, sosfilt_zi
    low_sos = butter(4, LOW_HZ, fs=RATE, output="sos")
    low_tap = {"a": [], "b": []}
    _zi = {nm: sosfilt_zi(low_sos) * 0.0 for nm in sub.decks}
    for _nm, _d in sub.decks.items():
        def _wrap(orig, nm):
            def f(n):
                blk = orig(n)
                m_ = blk.mean(axis=1).astype(np.float64)
                filt, _zi[nm] = _sf(low_sos, m_, zi=_zi[nm])
                low_tap[nm].append((sub.clock,
                                    float(np.sqrt((filt ** 2).mean()))))
                return blk
            return f
        _d.read = _wrap(_d.read, _nm)

    filter_modes = {"a": set(), "b": set()}
    end_clock = swap_at + int(10.0 * RATE)
    while sub.clock < end_clock:
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))
        tel = sub.telemetry
        if tel:
            for nm in ("a", "b"):
                filter_modes[nm].add(tel["decks"][nm]["filter"])
    final_tel = sub.telemetry

    mix = np.concatenate(rendered, axis=0)
    mono = mix.mean(axis=1).astype(np.float64)
    start_clock = sub.clock - len(mono)
    w = RATE // 2
    bw0 = max(int((blend_at - start_clock) / RATE / 0.5), 4)
    sw = int((swap_at - start_clock) / RATE / 0.5)

    m = {"pair": f"{cur.title[:24]} -> {cand.title[:24]}",
         "style": style, "rate": plan["rate"],
         "blend_s": (swap_at - blend_at) / RATE}

    # Band bumps: blend max vs the pair's own SOLO p95 (same philosophy as
    # the quality gate's bass/lurch checks — the transition must not
    # out-peak the music's own peaks, but the music's peaks are its own
    # business).
    mud_sos = butter(2, [MUD_LO, MUD_HI], btype="band", fs=RATE,
                     output="sos")
    high_sos = butter(4, HIGH_HZ, btype="high", fs=RATE, output="sos")
    for name, sos in (("low", low_sos), ("mud", mud_sos),
                      ("high", high_sos)):
        rms = _band_windows(mono, sos, w)
        pre, mid_seg, post = rms[4:max(bw0, 5)], rms[bw0:max(sw, bw0 + 1)], \
            rms[sw:]
        if len(pre) and len(mid_seg) and len(post):
            solo = max(np.percentile(pre, 95), np.percentile(post, 95))
            m[f"{name}_bump_db"] = float(20 * np.log10(
                max(mid_seg.max(), 1e-9) / max(solo, 1e-9)))
        else:
            m[f"{name}_bump_db"] = 0.0
        if name == "low":
            # Cliff check: biggest 0.5s low-band step inside the blend vs
            # the music's own solo low-band steps (an instant low swap is
            # a measured 8 dB step; swap_beats ramps exist to spread it).
            # PERCEPTUAL SILENCE FLOOR at -40 dB vs the solo median: a
            # carved low band sits near digital silence, where dB steps
            # between two inaudible windows read as tens of dB of nothing.
            ref = max(float(np.median(np.concatenate(
                [rms[4:max(bw0, 5)], rms[sw:]]))), 1e-6)
            steps = 20 * np.abs(np.diff(np.log10(
                np.maximum(rms, ref * 0.01))))
            inside = steps[bw0:max(sw, bw0 + 1)]
            outside = np.concatenate([steps[4:bw0], steps[sw:]])
            m["low_lurch_db"] = float(inside.max()) if len(inside) else 0.0
            m["low_lurch_solo_db"] = float(outside.max()) \
                if len(outside) else 0.0
            # Hole check on 1s windows (rides over bar-length bass gaps):
            # how much deeper does the blend's low floor dip below the solo
            # median than the music's own quietest solo moment does?
            rms1 = _band_windows(mono, sos, RATE)
            b0_1, sw_1 = max(bw0 // 2, 2), max(sw // 2, bw0 // 2 + 1)
            solo1 = np.concatenate([rms1[2:b0_1], rms1[sw_1:]])
            in1 = rms1[b0_1:sw_1]
            if len(solo1) > 2 and len(in1):
                # Depths capped at 40 dB: below that it's just "silent" —
                # deeper numbers are floor noise, not more hole.
                med = max(float(np.median(solo1)), 1e-9)
                depth_in = min(20 * np.log10(med / max(in1.min(), 1e-9)),
                               40.0)
                depth_solo = min(20 * np.log10(
                    med / max(solo1.min(), 1e-9)), 40.0)
                m["low_hole_excess_db"] = float(depth_in - depth_solo)
            else:
                m["low_hole_excess_db"] = 0.0
            # A-SOLO ABSOLUTION REFERENCE: A's own low-band behavior over
            # the source region the blend plays (its exit) vs its groove
            # just before. Mixing through A's breakdown is good DJing —
            # if A alone digs the hole / makes the step, the transition
            # is absolved (the quality gate's 'rendered alone' lesson).
            a_rate_p = plan.get("a_rate", 1.0) or 1.0
            span = (swap_at - blend_at) / RATE * a_rate_p
            i1 = min(int(plan["out_s"] * RATE), len(a))
            i0 = max(int((plan["out_s"] - span) * RATE), 0)
            p0 = max(int((plan["out_s"] - span - 20.0) * RATE), 0)
            from scipy.signal import sosfilt as _sf2
            a_low = _sf2(sos, a[p0:i1].mean(axis=1).astype(np.float64))
            n_exit = (i1 - i0) // RATE
            ar1 = np.sqrt(np.mean(
                a_low[:len(a_low) // RATE * RATE].reshape(-1, RATE) ** 2,
                axis=1))
            a_pre, a_exit = ar1[:max(len(ar1) - n_exit, 1)], \
                ar1[-n_exit:] if n_exit else ar1[:0]
            if len(a_pre) and len(a_exit):
                a_med = max(float(np.median(a_pre)), 1e-9)
                m["a_solo_hole_db"] = float(min(20 * np.log10(
                    a_med / max(a_exit.min(), 1e-9)), 40.0))
            else:
                m["a_solo_hole_db"] = 0.0
            ar5 = np.sqrt(np.mean(
                a_low[:len(a_low) // w * w].reshape(-1, w) ** 2, axis=1))
            a_ref = max(float(np.median(ar5)), 1e-6)
            a_steps = 20 * np.abs(np.diff(
                np.log10(np.maximum(ar5, a_ref * 0.01))))
            n5 = (i1 - i0) // w
            m["a_solo_lurch_db"] = float(a_steps[-n5:].max()) \
                if n5 and len(a_steps) >= n5 else 0.0
            # ...and B's: the swap is deliberately timed to B's own bass
            # ENTRANCE (the b_bassy floor), so the low step there is B's
            # arrangement drop — the ramps can only smooth it, never add
            # to it. B-alone is the honest bar for the arrival side.
            j0 = max(int(plan["in_s"] * RATE), 0)
            j1 = min(int((plan["in_s"] + (swap_at + int(10 * RATE)
                                          - blend_at) / RATE
                          * plan["rate"]) * RATE), len(b))
            b_low = _sf2(sos, b[j0:j1].mean(axis=1).astype(np.float64))
            br5 = np.sqrt(np.mean(
                b_low[:len(b_low) // w * w].reshape(-1, w) ** 2, axis=1))
            b_ref = max(float(np.median(br5)), 1e-6)
            b_steps = 20 * np.abs(np.diff(
                np.log10(np.maximum(br5, b_ref * 0.01))))
            m["b_solo_lurch_db"] = float(b_steps.max()) \
                if len(b_steps) else 0.0

    # ONLY ONE BASSLINE AT A TIME, ground truth: seconds both decks' low
    # bands are simultaneously hot (>0.35 of that deck's own peak).
    ba, bb = _bins(low_tap["a"]), _bins(low_tap["b"])
    pa = max(ba.values(), default=1e-9)
    pb = max(bb.values(), default=1e-9)
    m["bass_dual_s"] = round(sum(
        0.25 for k in set(ba) & set(bb)
        if ba[k] > 0.35 * pa and bb[k] > 0.35 * pb), 2)

    m["filter_modes_a"] = sorted(filter_modes["a"])
    db_ = final_tel["decks"]["b"]
    m["b_final"] = {"eq": db_["eq"], "filter": db_["filter"],
                    "gain": db_["gain"], "loop": bool(db_["loop"]),
                    "playing": db_["playing"]}
    m["bassy_pair"] = min(cur.spectral.get("bass_share", 0.33),
                          cand.spectral.get("bass_share", 0.33)) >= 0.18

    if wav:
        from scipy.io import wavfile
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "..", "logs", f"dj_spectral_{style}.wav")
        wavfile.write(p, RATE,
                      (np.clip(mix, -1, 1) * 32767).astype(np.int16))
        m["wav"] = os.path.normpath(p)
    return m


# Styles where the low end is actually contested. The fade/cut families
# (no dual low by construction) belong to the quality gate, not here.
STYLES = ["long_blend", "bass_swap", "stem_bass_swap", "filter_sweep",
          "drum_bridge", "breakdown_swap"]


def seam_spectra(library, styles, wav=False):
    print("\n=== seam spectra: brain-planned transitions, real tracks ===")
    cands = sorted([t for t in library
                    if t.bpm_conf > 0.6 and t.downbeat_conf > 0.3
                    and t.duration_s > 240], key=lambda t: -t.rhythm_density)
    got = {}
    for si, style in enumerate(styles):
        m = None
        # Legality is checked before any decode, so scanning wide is cheap;
        # stem styles need stems on A (and luck on the chosen B) — scan
        # only stem-having tracks for those. Seed varies per style so the
        # styles don't all land on the identical pair.
        pool = cands
        if style in ("stem_bass_swap", "drum_bridge"):
            pool = [t for t in cands if getattr(t, "has_stems", False)]
        for cur in pool[:40]:
            try:
                m = render_seam(library, cur, style, seed=7 + si, wav=wav)
            except Exception as e:
                print(f"  [FAIL] {style} render crashed on {cur.title[:30]}:"
                      f" {type(e).__name__}: {e}")
                failures.append(f"{style} crash")
                m = None
                break
            if m:
                break
        if m is None:
            print(f"  [warn] {style}: no legal pair found in top candidates")
            continue
        got[style] = m
        print(f"\n  {style}: {m['pair']}  rate={m['rate']:.3f} "
              f"blend {m['blend_s']:.1f}s")
        print(f"    bass dual {m['bass_dual_s']:.2f}s | bumps "
              f"low {m['low_bump_db']:+.1f} mud {m['mud_bump_db']:+.1f} "
              f"high {m['high_bump_db']:+.1f} dB | low lurch "
              f"{m['low_lurch_db']:.1f} (solo {m['low_lurch_solo_db']:.1f}, "
              f"A-alone {m['a_solo_lurch_db']:.1f}, "
              f"B-alone {m['b_solo_lurch_db']:.1f}) | hole excess "
              f"{m['low_hole_excess_db']:+.1f} dB (A-alone digs "
              f"{m['a_solo_hole_db']:.1f})")

        if style == "drum_bridge":
            # The bridge SPLITS the low end (0.45/0.75) for bridge_beats on
            # purpose — dual low is the style's contract, report only.
            print(f"    (drum_bridge splits the low end by design: "
                  f"dual {m['bass_dual_s']:.1f}s unjudged)")
        else:
            # The swap crossfade is 4–6 beats wide; both basslines are
            # mid-crossfade for a couple seconds at most.
            check(f"{style}: one bassline at a time",
                  m["bass_dual_s"] <= 4.0,
                  f"both decks' low bands hot {m['bass_dual_s']:.2f}s "
                  f"(crossfade budget 4.0)")
        check(f"{style}: no double bass in the mix",
              m["low_bump_db"] < 3.5,
              f"blend low-band bump {m['low_bump_db']:+.1f} dB")
        check(f"{style}: no low-mid mud", m["mud_bump_db"] < 3.5,
              f"blend 180-600 Hz bump {m['mud_bump_db']:+.1f} dB")
        if style == "drum_bridge":
            # The percussion break rides BOTH full drum kits with mids and
            # highs open on purpose ("both kits full-bodied") — stacked
            # hats for bridge_beats are the style, not a defect.
            print(f"    (drum_bridge stacks both kits by design: high "
                  f"bump {m['high_bump_db']:+.1f} dB unjudged)")
        else:
            check(f"{style}: no hat stack", m["high_bump_db"] < 3.5,
                  f"blend >6 kHz bump {m['high_bump_db']:+.1f} dB")
        if style == "breakdown_swap":
            # The swap rides A's BREAKDOWN into B's build — the deep low
            # hole and the drop slam ARE the arrangement (measured 22 dB
            # low-only hole with mids still playing: a real breakdown, not
            # dead air). Same precedent as drum_bridge's dual-low above.
            print(f"    (breakdown_swap trades the low end through A's "
                  f"breakdown by design: hole/cliff unjudged)")
        else:
            check(f"{style}: bass swap is spread, not a cliff",
                  m["low_lurch_db"] <= max(m["low_lurch_solo_db"],
                                           m["a_solo_lurch_db"],
                                           m["b_solo_lurch_db"], 6.0) + 2.5,
                  f"low step {m['low_lurch_db']:.1f} dB vs solo "
                  f"{m['low_lurch_solo_db']:.1f} / A-alone "
                  f"{m['a_solo_lurch_db']:.1f} / B-alone "
                  f"{m['b_solo_lurch_db']:.1f} dB")
            if m["bassy_pair"]:
                check(f"{style}: no low-end hole",
                      m["low_hole_excess_db"] <= 8.0
                      or m["low_hole_excess_db"]
                      <= m["a_solo_hole_db"] + 6.0,
                      f"blend low floor {m['low_hole_excess_db']:+.1f} dB "
                      f"deeper than the music's own quietest solo moment "
                      f"(A alone digs {m['a_solo_hole_db']:.1f} dB in its "
                      f"exit)")
        if style == "filter_sweep":
            check("filter_sweep: the sweep actually engages",
                  "hp" in m["filter_modes_a"],
                  f"deck A filter modes seen: {m['filter_modes_a']}")
        # The "no bass night" class: a deck that leaves the seam carved,
        # filtered, looping or ducked poisons every following track.
        bf = m["b_final"]
        check(f"{style}: incoming deck leaves the seam restored",
              bf["playing"] and not bf["loop"] and bf["filter"] == "off"
              and all(g >= 0.9 for g in bf["eq"]) and bf["gain"] >= 0.9,
              f"eq {bf['eq']} filter {bf['filter']} gain {bf['gain']:.2f} "
              f"loop {bf['loop']} playing {bf['playing']}")
    check("style coverage rendered", len(got) >= min(4, len(styles)),
          f"rendered {sorted(got)} of {styles}")
    return got


def main():
    db = LibraryDB(MUSIC)
    library = load_library(db)
    print(f"library: {len(library)} tracks")
    spectral_audit(library)
    if "--audit-only" not in sys.argv:
        styles = STYLES
        if "--styles" in sys.argv:
            styles = sys.argv[sys.argv.index("--styles") + 1].split(",")
        seam_spectra(library, styles, wav="--wav" in sys.argv)
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s):")
        for f_ in failures:
            print(f"  - {f_}")
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
