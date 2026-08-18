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

def persona_bias_audit(reachable):
    """Every persona's SIGNATURE must point at a style that can be played.

    This has silently broken twice. showman's style_bias was loop_build
    for nine days after that style was retired, and crate_digger's was
    loop_roll_exit for the same nine - so both personas' defining move
    multiplied a weight kill() zeroed on every seam, and both still LOOKED
    fine because their other levers carried them. A style_bias naming a
    dead style is not a small bug: it is the persona's whole identity
    quietly doing nothing, and nothing else in the engine notices.

    `reachable` is the set of styles the selection audit saw ON A MENU -
    a reachability claim, not a luck claim, so a legitimately rare style
    still counts as long as the engine offered it at least once.
    """
    from lib.dj.persona import PERSONAS
    dead = []
    for p in PERSONAS.values():
        for style in (p.style_bias or {}):
            if style not in reachable:
                dead.append(f"{p.name}:{style}")
    check("persona signatures point at playable styles", not dead,
          f"dead style_bias targets: {dead}" if dead
          else f"all {sum(len(p.style_bias or {}) for p in PERSONAS.values())}"
               f" targets reachable")


def selection_audit(library, theme):
    print("\n=== selection audit: brain choice from every library track ===")
    styles, rates, scores = {}, [], []
    reachable = set()
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
        # What the DICE were offered, not just what won - a style biased by
        # a persona needs only to be reachable for the bias to mean
        # something.
        reachable.update((plan.get("diag") or {}).get("menu") or {})
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
    persona_bias_audit(reachable)
    # Stretch discipline. The wall moved 8% -> 10% (2026-08-06, operator's
    # call) and the 5.5% selection cliff now only applies to UNVERIFIED
    # grids, so the distribution deliberately widened: measured on the
    # real library the same day, median 1.1% / p95 7.4% / max 10.0%
    # (was median 1.1% / p95 4.7% / max 7.4%).
    # This is still a real gate, on two things that must never happen:
    #   - MAX beyond the wall. 10.1% catches a plan asking for a rate
    #     deck.set_rate would silently clamp (np.clip 0.90..1.10), which
    #     is how a tempo bug hides - the deck obeys, the seam drifts.
    #   - the TYPICAL seam creeping. Median stays tight; most of a night
    #     is still near-native, the wide band is the exception the wall
    #     exists to allow, not the new normal.
    check("stretch discipline",
          np.median(rates) <= 0.03 and np.percentile(rates, 95) <= 0.085
          and max(rates) <= 0.101,
          f"median {np.median(rates)*100:.1f}% p95 "
          f"{np.percentile(rates, 95)*100:.1f}% max {max(rates)*100:.1f}% "
          f"(wall 10%, verified-only past 5.5%)")
    # REPERTOIRE GUARDS (rewritten 2026-08-05). The old check capped the
    # fade share - but the user's taste decisions now ROUTE most
    # unmixable pairs to the deliberate fade (echo demoted to
    # punctuation: "you are way overusing echo out"), so a high fade
    # share is policy, not failure. What must never regress:
    # 1) BLENDS COLLAPSING - the trapdoor bug read as exactly that
    #    (blend family ~5% because gated pairs leaked into fake
    #    bass_swaps). The blend family must stay a real presence.
    # 2) ECHO DOMINANCE - echo_out winning blend-less menus by default
    #    was 40% of the night.
    blend_n = sum(styles.get(k, 0) for k in
                  ("long_blend", "bass_swap", "filter_sweep",
                   "stem_bass_swap", "melody_carry"))
    check("blend family is a real presence", blend_n >= n * 0.08,
          f"{blend_n}/{n} seams plan an overlapped blend "
          f"({blend_n/max(n,1)*100:.0f}%; floor 8%)")
    check("echo stays punctuation", styles.get("echo_out", 0) <= n * 0.25,
          f"{styles.get('echo_out', 0)}/{n} seams take echo_out "
          f"({styles.get('echo_out', 0)/max(n,1)*100:.0f}%; cap 25%)")
    check("style variety actually used", len(styles) >= 4,
          f"{len(styles)} distinct styles chosen: {sorted(styles)}")
    check("pair scores not collapsed", np.median(scores) > 0.05,
          f"median pair score {np.median(scores):.3f}")
    return styles


# ==========================================================================
# Part 2: seam renders
# ==========================================================================

def force_style(theme, style):
    # Styles ABSENT from a theme's weight dict get accent-tier defaults in
    # plan_transition — zeroing only the dict's existing keys leaves those
    # alive and the forced style has to win a dice roll against them.
    # Enumerate the defaulted vocabulary explicitly so forcing is forcing.
    # sorted(): THIS WAS THE HASHSEED LEAK (2026-08-14). Iterating the
    # raw SET put the weights dict — and therefore plan_transition's
    # menu list — in per-process hash order, so the seeded style dice
    # drew a DIFFERENT element under different PYTHONHASHSEEDs: the
    # same Calexico -> Tarantula search planned breakdown_swap under
    # hash seeds 5/17 and long_fade under 0/11/23/42, which is exactly
    # the "render nondeterminism, source not yet found" (2026-08-05)
    # and this gate's intermittent coverage misses. Sets never feed
    # ordered structures that meet an RNG.
    # COPY, never the shared instance (2026-08-17). get_theme returns the
    # BUILTIN_THEMES singleton, and rebinding style_weights on it pinned
    # the style for EVERY later get_theme() in the process - the audible
    # calibration's "natural" picks all rolled a one-style menu after the
    # first forced render (operator caught it: "thats a lot of flam" on
    # a 60% flam bucket that was secretly all kit styles).
    import copy
    t = copy.copy(get_theme(theme.name))
    known = set(t.style_weights) | {
        "stem_drum_swap", "acapella_out", "stem_bass_swap", "drum_bridge",
        "acapella_in", "melody_carry", "phrase_cut",
        "breakdown_swap"}
    t.style_weights = {k: (1.0 if k == style else 0.0)
                       for k in sorted(known)}
    return t


def render_seam(library, cur, style, wav=False, allow_benched=False,
                pair=None, b_veto=None, tune=None, decoded=None,
                test_gates=False, gap_policy=True):
    """Arm one brain-planned transition exactly like DJSystem does and
    render it offline. Returns (metrics dict | None if style not legal).

    `allow_benched` is for AUDITION PROBES only - it lets a style that is
    off the live menu pending a listen be rendered and measured. The gate
    below never passes it: a benched style must not be able to satisfy the
    gate's own coverage check.

    `b_veto` narrows choose_next's B side to structurally viable partners
    (audition_pools' veto set, the same aid the Lab uses) - a SEARCH aid,
    never a gate: every screen still runs on whatever gets picked."""
    theme = force_style(get_theme("groove"), style)
    brain = Brain(library, theme, seed=7)
    if b_veto:
        brain.veto_ids = set(b_veto)
    brain.note_played(cur)
    if pair is not None:
        # AUDITION PROBES supply their own (cand, meta): a style rare enough
        # that this function's own choose_next never lands it cannot be
        # measured otherwise, and "too rare to sample" must not be
        # indistinguishable from "sounds fine".
        cand, meta = pair
    else:
        cand, meta = brain.choose_next(cur, 0.6, cur.bpm)
    if cand is None:
        return None
    # force_style PIN, not just themed weights (2026-08-14): the themed
    # weights still leave long_fade's 0.8 dice presence on every menu,
    # so a LEGAL plan for the style under test lost ~45% of its rolls -
    # and which rolls it lost depended on the menu's dict order, which
    # force_style used to build from a raw SET. That pair of facts was
    # this gate's intermittent style-coverage misses (and the wider
    # "render nondeterminism"): hash order flipped whether the one
    # findable pair won its roll. The pin removes the dice entirely -
    # safety gates still outrank it, so an illegal pair still refuses.
    plan = brain.plan_transition(cur, cand, meta,
                                 after_s=cur.duration_s * 0.45,
                                 force_style=style,
                                 allow_benched=allow_benched,
                                 test_gates=test_gates)
    if plan["style"] != style:
        return None                       # gates said no (no drop/loop/...)
    if tune:
        # AUDITION PROBES ONLY: per-seam knob overrides, the same K()
        # channel the Lab's jitter uses - lets an A/B render the same
        # seam with a knob on vs off. The gate itself never passes this.
        plan["tune"] = dict(tune)

    # `decoded` is a caller-managed {track_id: samples} cache: the night
    # simulator chains seams (B becomes the next A), so each track needs
    # decoding once, not twice. Caller prunes it - full tracks are
    # ~100MB each.
    def _dec(t):
        if decoded is not None and t.id in decoded:
            return decoded[t.id]
        arr = F.decode_file_stereo(os.path.join(MUSIC, t.path))
        if decoded is not None:
            decoded[t.id] = arr
        return arr
    a = _dec(cur)
    b = _dec(cand)

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
              "track_id": cur.id, "gain_db": cur.gain_db, "cue_s": cue_a})
    sub.post({"cmd": "gain", "deck": "a", "value": 1.0, "ramp_s": 0.01})
    sub.post({"cmd": "start", "deck": "a"})

    gen = engine._mixer()
    next(gen)
    rendered = []
    for _ in range(int(2.0 * RATE) // BLOCK):     # telemetry warm-up
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))

    sub.post({"cmd": "load", "deck": "b", "samples": b, "grid": cand.grid,
              "track_id": cand.id, "gain_db": cand.gain_db,
              "cue_s": plan["in_s"]})
    # STEM STYLES RENDER WITH STEMS (2026-08-17). This harness never
    # attached them, so every stem-style render played stem_gains
    # against a stem-less deck (a no-op): the measured audio was the EQ
    # carve on full mixes, not what the live seam plays. Mirror
    # audition.py - attach when the style (or a vocal duck) needs them.
    _need_stems = plan["style"] in (
        "stem_drum_swap", "drum_bridge", "stem_bass_swap",
        "acapella_out", "acapella_in", "melody_carry") \
        or plan.get("duck_vocal_a")
    if _need_stems:
        from lib.dj.stems import load_stems
        for deck, t, arr in (("a", cur, a), ("b", cand, b)):
            if getattr(t, "has_stems", False):
                st_ = load_stems(MUSIC, t.id, expected_len=len(arr))
                if st_:
                    sub.post({"cmd": "attach_stems", "deck": deck,
                              "stems": st_})
    for _ in range(4):
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))
    if gap_policy:
        # The shared predicted-exposure dip policy (lib/dj/gapscan) -
        # the SAME call the live arm path makes, so harness renders
        # keep predicting the night.
        from lib.dj import gapscan as _GS
        plan, events, swap_at, blend_at, _gap_act = _GS.apply_gap_policy(
            brain, plan, cur, cand, meta, a, b, sub.telemetry, "a", "b",
            after_s=cur.duration_s * 0.45)
        style = plan["style"]
    else:
        events, swap_at, blend_at = brain.build_events(
            plan, sub.telemetry, "a", "b", cur, cand)
    sub.post_many(events)

    from lib.dj.deck import ENV_FPS
    beat = cur.period_s
    end_clock = swap_at + int(10.0 * RATE)
    lags, grid_lags, dual = [], [], 0.0
    biases = []
    tel_log = []
    # The LIVE audible meter's readings (sync.audible_err_beats), sampled
    # at its own ~4Hz cadence AND its own settled window (blend+6s -
    # mirroring system._collect_seam_metrics exactly; the first batch
    # collected from the first dual block and inflated flags with PLL
    # convergence readings the live collector never counts) - so an
    # offline batch can calibrate the meter against this harness's
    # env-xcorr ground truth (see _dj_audible_calib.py). aud_series
    # keeps every settled sample so threshold sweeps re-run offline
    # without re-rendering.
    aud_max, aud_n, _aud_last = 0.0, 0, None
    aud_series = []
    # Per-deck (render clock, source seconds) traces - what projects
    # each deck's grid into render time for the isolated kick
    # measurement (seamverify.measured_kick_alignment).
    pos_trace = {"a": [], "b": []}
    # PER-DECK MID-BAND TAP (250-2500 Hz - where melodies live): wrap each
    # deck's read so we can verify ONE MELODY AT A TIME in the actual
    # rendered audio, not just in the scheduled events.
    from scipy.signal import butter as _butter, sosfilt as _sf, sosfilt_zi
    _mid_sos = _butter(2, [250.0, 2500.0], btype="band", fs=RATE,
                       output="sos")
    mid_tap = {"a": [], "b": []}
    # ...and the raw post-EQ/post-gain mono, CLOCK-PLACED (the mixer skips
    # read() for a deck that is not playing, so plain appending would
    # time-shift the late-starting deck to the render start). This is what
    # perc_overlap needs: it must see the band carves and the fader.
    deck_pcm = {"a": [], "b": []}
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
                deck_pcm[nm].append(
                    (sub.clock, blk.mean(axis=1).astype(np.float32)))
                return blk
            return f
        _d.read = _wrap(_d.read, _nm)
    i = len(rendered)
    while sub.clock < end_clock:
        rendered.append(np.frombuffer(gen.send(BLOCK),
                                      np.float32).reshape(-1, 2))
        i += 1
        if i % 4 == 0:
            # Both stamps are end-of-block (sub.clock has advanced and
            # source_time_s is the cursor), so the pairing is exact.
            for _nm in ("a", "b"):
                _d = sub.decks[_nm]
                if _d.playing:
                    pos_trace[_nm].append((sub.clock,
                                           _d.source_time_s()))
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
            _ae = (tel.get("sync") or {}).get("audible_err_beats")
            if _ae is not None \
                    and tel["clock"] >= blend_at + int(6.0 * RATE):
                _a = abs(float(_ae))
                aud_max = max(aud_max, _a)
                if _aud_last is None or tel["clock"] - _aud_last >= RATE // 4:
                    aud_series.append((round(dual, 2), round(_a, 4)))
                    if _a > 0.12:
                        aud_n += 1
                    _aud_last = tel["clock"]
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
                if sy:
                    biases.append(float(sy.get("bias_beats") or 0.0))
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
         "gap": plan.get("gap"),
         "fade_reason": (plan.get("diag") or {}).get("fade_reason"),
         "in_s": round(plan.get("in_s", 0.0), 2),
         "out_s": round(plan.get("out_s", 0.0), 2),
         "dual_s": dual, "n_lags": len(settled),
         "lag_early": float(np.median(early)) if early else None,
         "lag_med": float(np.median(settled)) if settled else None,
         "lag_max": float(np.max(settled)) if settled else None,
         "aud_max": round(aud_max, 4), "aud_n": aud_n,
         "aud_series": aud_series,
         "raw_lags": lags, "grid_lags": grid_lags,
         "peak": float(np.abs(mix).max()),
         "clipped": int((np.abs(mix) > 0.999).sum())}
    # KICK-BIAS WIRING TRAP (2026-08-04): sync must offset the slave grid
    # by exactly (slave music-phase - master music-phase), as computed by
    # the brain and shipped in the sync event. A sign flip or a dropped
    # event field DOUBLES the audible kick error while every grid metric
    # stays green - assert the telemetry bias against the plan's record.
    _pa = plan.get("phase_applied") or {}
    m["bias_seen"] = float(np.median(biases)) if biases else None
    m["bias_expected"] = (
        None if not _pa
        else float(np.clip(
            _pa["b_ms"] / 1000.0 / max(cand.period_s, 1e-6)
            - _pa["a_ms"] / 1000.0 / max(cur.period_s, 1e-6),
            -0.25, 0.25)))    # mirror the brain's clip

    # ISOLATED-KICK ALIGNMENT (2026-08-17): the instrument that replaces
    # the env-xcorr lag as ground truth - per-deck kick clocks, never a
    # cross-correlation (see seamverify.measured_kick_alignment for the
    # ear-exam failure that forced this).
    _sc0 = sub.clock - len(mono)
    try:
        from lib.dj import seamverify as _SV
        _deck_arr = {}
        for _nm in ("a", "b"):
            _arr = np.zeros(len(mono), dtype=np.float32)
            for _clk, _blk in deck_pcm[_nm]:
                _i0 = int(_clk - _sc0)
                if 0 <= _i0 and _i0 + len(_blk) <= len(_arr):
                    _arr[_i0:_i0 + len(_blk)] = _blk
            _deck_arr[_nm] = _arr
        _marks = {"blend_s": (blend_at - _sc0) / RATE,
                  "swap_s": (swap_at - _sc0) / RATE,
                  "pos": {nm: [((c - _sc0) / RATE, s)
                               for c, s in pos_trace[nm]]
                          for nm in ("a", "b")}}
        m["kick_iso"] = _SV.measured_kick_alignment(
            _deck_arr, _marks, cur, cand)
    except Exception as _e:
        m["kick_iso"] = None
        m["kick_iso_err"] = str(_e)

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

    # ONE KICK AT A TIME: mid_overlap_s above asks the question on LEVEL,
    # which is the right question for melodies. Percussion needs the same
    # question asked on TRANSIENTS - and on a long_fade it is the ONLY
    # rhythm question that means anything, because the decks were never
    # matched (no sync event -> max_err_beats is 0.000 on every fade this
    # repo has ever logged, so 160 of 165 logged 'clean' while the user
    # was hearing kick clash).
    from lib.dj.seamverify import perc_overlap

    def _scatter(blocks):
        buf = np.zeros(len(mono), dtype=np.float32)
        for c, blk in blocks:
            i = c - start_clock
            if i < 0 or i >= len(buf):
                continue
            j = min(i + len(blk), len(buf))
            buf[i:j] = blk[:j - i]
        return buf
    _po = perc_overlap(_scatter(deck_pcm["a"]), _scatter(deck_pcm["b"]),
                       (blend_at - start_clock) / RATE,
                       (swap_at - start_clock) / RATE + 6.0)
    m["perc_kick_s"] = _po["kick_s"]
    m["perc_hi_s"] = _po["perc_s"]

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
    # (double_drop + bassline_layer removed 2026-08-02; the newer cut/loop
    # entries joined the render pool. cut_at_drop REJOINED 2026-08-12 when
    # it came off the bench - it is the only style that hard-cuts with no
    # overlap, so it is the one whose grid lock nothing else can cover for,
    # and it now also carries the outgoing tempo ramp: if the meet-in-the-
    # middle ever mistimes, the cut misses the drop and this render is what
    # notices.)
    # (loop family + spinback_cut retired 2026-08-04, user verdict on
    # the roll/slowdown mechanics.)
    # breakdown_swap benched 2026-08-04 (drop/EQ-restore stacking slam),
    # rebuilt 2026-08-13 (entry takes the build whose drop ARRIVES, the
    # restore clears it; lurch 3.3 dB median), un-benched 2026-08-14
    # after 12/14 good in the Lab - so it renders here again: the
    # restore-vs-drop timing is exactly what this gate measures.
    # phrase_cut retired 2026-08-05 ("I have never heard a good phrase
    # cut" - user); echo_out is the remaining cut style.
    # long_fade is rendered TWICE, because it has two populations and
    # only one of them was ever tested. The conf<0.45 pool is the
    # `grid_conf<0.5` fade: quiet, often beatless, and it passes every
    # check trivially. But 88% of armed fades carry no fade_reason at all
    # - they won the dice after the blends were gated, on RHYTHMIC pairs
    # (median kick_agreement 0.874 over the logged nights). That is where
    # "awful kick clashes" live, so it gets its own render off the
    # beat-heavy pool.
    styles = ["bass_swap", "long_blend", "filter_sweep", "echo_out",
              "cut_at_drop", "breakdown_swap", "long_fade",
              "long_fade_beaty"]
    fade_cands = sorted([t for t in library
                         if t.bpm_conf < 0.45 and t.duration_s > 240],
                        key=lambda t: -t.rhythm_density)
    got = {}
    for label in styles:
        m = None
        # `label` names the CASE, `style` the style actually planned.
        style = "long_fade" if label.startswith("long_fade") else label
        pool = fade_cands if label == "long_fade" else cands
        # SEARCH DEPTH FOLLOWS HOW RARE THE STYLE'S GATES MAKE IT. Twelve
        # candidates is plenty for the workhorses, and finds nothing at all
        # for cut_at_drop: it is the only style requiring bpm_conf>=0.8 on
        # BOTH sides AND a pre_drop entry in B, which leaves ~15% of pairs
        # structurally eligible. At 12 it silently reported "no legal pair"
        # and the coverage check passed anyway - the style was in the list
        # and never actually rendered. breakdown_swap got the same
        # treatment 2026-08-14 the day it was un-benched (same silent
        # warn on its first gated run), PLUS the audition_pools A-side
        # aid: its A must own a breakdown section, which a
        # rhythm-density-sorted pool has no reason to surface in 90
        # tries. The pools are a search aid, never a gate - render_seam
        # still runs every screen (and imports the pool rules from the
        # engine, so they cannot drift).
        depth = 90 if label in ("cut_at_drop", "breakdown_swap") else 12
        pool_l, b_veto = pool, None
        if label == "breakdown_swap":
            from lib.dj.brain import audition_pools
            a_pool, b_veto = audition_pools(library, style)
            ids = {t.id for t in a_pool}
            pool_l = [t for t in pool if t.id in ids] or pool
        for cur in pool_l[:depth]:
            try:
                m = render_seam(library, cur, style, wav=wav, b_veto=b_veto)
            except Exception as e:
                print(f"  [FAIL] {label} render crashed on {cur.title[:30]}: "
                      f"{type(e).__name__}: {e}")
                failures.append(f"{label} crash")
                m = None
                break
            if m:
                # KNOWN FLAKE (2026-08-05): the grid-lock measurement on
                # borderline material flips pass/fail across identical
                # renders (The Heck -> Slowdive: 4ms, 4ms, then 150ms+ at
                # the same config - render nondeterminism, source not yet
                # found). One re-render on a failing lock keeps the gate
                # honest about real regressions without letting a coin
                # flip block the pipeline; the flake is PRINTED so it
                # cannot hide.
                gl = [l for d, l in m["grid_lags"] if d > 2.0]
                med_bar = 35.0 if style in ("echo_out", "phrase_cut",
                                            "spinback_cut") else 25.0
                if gl and style != "long_fade" \
                        and float(np.median(gl)) > med_bar:
                    print(f"  [FLAKY?] {label}: grid med "
                          f"{np.median(gl):.0f}ms - re-rendering once")
                    m2 = render_seam(library, cur, style, wav=wav,
                                     b_veto=b_veto)
                    if m2:
                        gl2 = [l for d, l in m2["grid_lags"] if d > 2.0]
                        if gl2 and np.median(gl2) < np.median(gl):
                            print(f"  [FLAKY!] retry measured "
                                  f"{np.median(gl2):.0f}ms - keeping it")
                            m = m2
                break
        if m is None:
            print(f"  [warn] {label}: no legal pair found in top candidates")
            continue
        got[label] = m
        lag = (f"lag settled med {m['lag_med']:.0f}ms max {m['lag_max']:.0f}ms"
               if m["lag_med"] is not None else "no settled dual window")
        if m["lag_early"] is not None:
            lag += f" (launch {m['lag_early']:.0f}ms)"
        print(f"\n  {label}: {m['pair']}  rate={m['rate']:.3f}")
        if "worst_steps" in m:
            print(f"    worst steps (s after blend start, dB): "
                  f"{m['worst_steps']}  swap at +{m['swap_rel_s']}s")
            print(f"    events: {m['events']}")
        print(f"    dual {m['dual_s']:.1f}s | {lag} | peak {m['peak']:.2f} "
              f"clip {m['clipped']} | rms_min {m['rms_min_ratio']:.2f} "
              f"lurch {m['lurch_db']:.1f}dB (solo {m['lurch_solo_db']:.1f}) "
              f"| bass bump {m['bass_bump_db']:+.1f}dB")
        print(f"    both decks live: kick {m['perc_kick_s']:.2f}s "
              f"transient {m['perc_hi_s']:.2f}s "
              f"| mid {m['mid_overlap_s']:.2f}s")
        # The DIP styles breathe at the seam by contract: long_fade's
        # dipped handoff and echo_out's cut-into-decaying-tail both let
        # the room drop to ~-17 dB for a window on quiet-intro pairs.
        # Their bar is 0.10 (-20 dB): still strictly above the v2 fade's
        # -22 dB hole that was killed as 'stays dead', while a flat 0.15
        # failed legitimate dips as the render pair rotates.
        da_bar = 0.10 if style in ("long_fade", "echo_out") else 0.15
        check(f"{label}: no dead air", m["rms_min_ratio"] > da_bar,
              f"min/median RMS {m['rms_min_ratio']:.2f} (bar {da_bar})")
        check(f"{label}: no unmusical lurch",
              m["lurch_db"] <= max(m["lurch_solo_db"], 4.0) + 2.5,
              f"blend step {m['lurch_db']:.1f} dB vs solo "
              f"{m['lurch_solo_db']:.1f} dB")
        check(f"{label}: no clipping", m["clipped"] == 0
              and m["peak"] <= 1.0, f"peak {m['peak']:.3f} "
              f"clipped {m['clipped']}")
        check(f"{label}: no double bass", m["bass_bump_db"] < 3.5,
              f"blend low-band bump {m['bass_bump_db']:+.1f} dB")
        if style == "long_fade":
            # The dipped handoff: the two songs may BOTH be loud for only
            # a moment - a 12s full-range wash on an unmixable pair was
            # exactly what 'terribly mixed' sounded like.
            check(f"{label}: overlap is a dip, not a wash",
                  m["mid_overlap_s"] <= 3.5,
                  f"both mid-bands hot {m['mid_overlap_s']:.1f}s "
                  f"(dip budget 3.5)")
            # ONE KICK AT A TIME. The fade's decks are unsynced by
            # design, so two live kick patterns cannot be "tightened" -
            # they must not coexist. The low band is handed over as a
            # baton at the seam instead of the two ramps crossing.
            # BUDGET FROM MEASUREMENT, not from taste: over 10 rhythmic
            # pairs A/B'd against the old crossing ramps, co-presence
            # went mean 1.07s -> 0.40s (worst 2.75 -> 1.25). 1.5s sits
            # above every measured post-fix seam and below the bad
            # pre-fix ones, so it catches a regression to crossing ramps
            # without flaking on the residual crossfade window. Renders
            # here are not deterministic (+-0.25s on the same pair), so
            # a tighter bar would be a coin flip, not a gate.
            check(f"{label}: one kick at a time",
                  m["perc_kick_s"] <= 1.5,
                  f"both kick bands live {m['perc_kick_s']:.2f}s "
                  f"(budget 1.5)")
        else:
            # One melody at a time.
            check(f"{label}: one melody at a time",
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
            med_bar = 35.0 if style in ("echo_out", "phrase_cut",
                                        "spinback_cut") else 25.0
            check(f"{label}: decks grid-locked",
                  float(np.median(gl)) <= med_bar
                  and float(np.percentile(gl, 95)) <= 60.0,
                  f"grid delta med {np.median(gl):.0f}ms "
                  f"p95 {np.percentile(gl, 95):.0f}ms "
                  f"(harsh: {med_bar:.0f}/60)")
        if (m.get("bias_expected") is not None
                and m.get("bias_seen") is not None
                and style != "long_fade"):
            check(f"{label}: kick bias wired (sign + value)",
                  abs(m["bias_seen"] - m["bias_expected"]) <= 0.02,
                  f"seen {m['bias_seen']:+.4f} beats vs expected "
                  f"{m['bias_expected']:+.4f} (music-phase diff)")
        min_dual = {"bass_swap": 4.0, "long_blend": 4.0, "loop_in": 4.0,
                    "breakdown_swap": 4.0, "phrase_cut": 3.0,
                    "spinback_cut": 3.0}.get(style)
        if min_dual and m["dual_s"] < min_dual:
            check(f"{label}: decks actually overlap", False,
                  f"dual-audible only {m['dual_s']:.1f}s "
                  f"(need {min_dual:.0f})")
    # EVERY listed style must actually render - "no legal pair" plus a
    # >=4 floor let cut_at_drop (2026-08-13) and then breakdown_swap
    # (2026-08-14, the day it was un-benched) sit in the list and never
    # be tested.
    # (The intermittent misses this check used to show - breakdown_swap
    # 1-run-in-3, echo_out at depth 12 - were BOTH the hashseed dice
    # bug: force_style() iterated a raw SET into the weights dict, so a
    # LEGAL plan's menu order followed per-process hash order and the
    # seeded roll could lose to long_fade's 0.8 floor. Fixed 2026-08-14
    # evening: sorted(known) + render_seam pins the style, no roll at
    # all. echo_out's exemption went with it - every listed style is
    # required now, and a miss is a real finding, not a flake.)
    required = set(styles)
    check("style coverage rendered", required <= set(got),
          f"rendered {sorted(got)} of {styles}"
          + (f" - MISSING {sorted(required - set(got))}"
             if not required <= set(got) else ""))
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
