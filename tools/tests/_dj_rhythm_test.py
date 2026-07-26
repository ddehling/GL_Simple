"""Tests for lib/dj/rhythm.py: signature extraction on synthetic band
matrices with KNOWN patterns, pairwise terms, tempo-multiple recovery,
JSON roundtrip, and the chips vocabulary.

Run: python tools/tests/_dj_rhythm_test.py
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np

from lib.dj.rhythm import (N_STEPS, prep_signature, region_view,
                           rhythm_signature, rhythm_terms, seam_chips,
                           seam_rhythm, tempo_mult_for)

FPS = 40
CHECKS = {"pass": 0, "fail": 0}


def check(name, cond, detail=""):
    ok = bool(cond)
    CHECKS["pass" if ok else "fail"] += 1
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f"  ({detail})"
                                                    if detail and not ok else ""))


def synth_bands(bpm=120.0, dur_s=120.0, kick_16ths=(0, 4, 8, 12),
                snare_16ths=(4, 12), hat_phase=0.5, hat_on_beat=True):
    """Band matrix [T,32] with impulses at chosen 16th-note steps (per bar)
    plus an offbeat-8th hat at `hat_phase` within every beat."""
    T = int(dur_s * FPS)
    rng = np.random.RandomState(7)
    bands = 0.05 + 0.01 * rng.rand(T, 32)
    period = 60.0 / bpm
    step = period / 4.0
    t = 0.0
    while t < dur_s - 1.0:
        bar_start = t
        for s16 in range(16):
            ts = bar_start + s16 * step
            f = int(ts * FPS)
            if f + 2 >= T:
                break
            if s16 in kick_16ths:
                bands[f:f + 2, 0:6] += 6.0
            if s16 in snare_16ths:
                bands[f:f + 2, 6:16] += 5.0
        for beat in range(4):
            bt = bar_start + beat * period
            if hat_on_beat:
                f = int(bt * FPS)
                if f + 2 < T:
                    bands[f:f + 2, 16:32] += 3.0
            f = int((bt + hat_phase * period) * FPS)
            if f + 2 < T:
                bands[f:f + 2, 16:32] += 3.0
        t += 4 * period
    return bands


def grid_for(bpm, dur_s):
    return [{"start_s": 0.0, "end_s": dur_s, "period_s": 60.0 / bpm,
             "first_beat_s": 0.0, "bpm": bpm, "score": 1.0}]


print("== signature extraction ==")
BPM = 120.0
four = rhythm_signature(synth_bands(BPM), grid_for(BPM, 120.0), 0, fps=FPS, latency_s=0.0)
check("signature computed", four is not None)
sig4 = prep_signature(json.loads(json.dumps(four)))     # JSON roundtrip
check("roundtrip hydrates", sig4 is not None and len(sig4["low"]) == N_STEPS)

on_steps = [i for i in range(N_STEPS) if i % 4 == 0]
off_steps = [i for i in range(N_STEPS) if i % 4 != 0]
check("kicks land on the beat steps",
      min(sig4["low"][i] for i in on_steps) > 0.6,
      f"min on-step {min(sig4['low'][i] for i in on_steps):.2f}")
check("no kicks between beats",
      max(sig4["low"][i] for i in off_steps) < 0.35,
      f"max off-step {max(sig4['low'][i] for i in off_steps):.2f}")
check("straight hats -> swing ~0.5", abs(sig4["swing"] - 0.5) < 0.045,
      f"swing {sig4['swing']}")

swung = rhythm_signature(synth_bands(BPM, hat_phase=0.63),
                         grid_for(BPM, 120.0), 0, fps=FPS, latency_s=0.0)
swungp = prep_signature(swung)
check("swung hats detected", 0.56 <= swungp["swing"] <= 0.70,
      f"swing {swungp['swing']}")
check("swing confident", swungp["swing_conf"] > 0.3,
      f"conf {swungp['swing_conf']}")

offbeat = prep_signature(rhythm_signature(
    synth_bands(BPM, kick_16ths=(2, 6, 10, 14)), grid_for(BPM, 120.0), 0,
    fps=FPS, latency_s=0.0))
check("offbeat kicks land off the beat",
      max(offbeat["low"][i] for i in on_steps) < 0.35)

print("== pairwise terms ==")
period = 60.0 / BPM
same = rhythm_terms(sig4, sig4, 1.0, period)
check("identical grooves agree", same["kick_agreement"] > 0.9,
      f"agr {same['kick_agreement']}")
check("identical grooves score high", same["score"] > 0.8,
      f"score {same['score']}")

clash = rhythm_terms(sig4, offbeat, 1.0, period)
check("on-beat vs offbeat kicks clash", clash["kick_agreement"] < 0.4,
      f"agr {clash['kick_agreement']}")
check("clash scores below match", clash["score"] < same["score"] - 0.15)

sw = rhythm_terms(sig4, swungp, 1.0, period)
check("straight vs swung -> swing_delta", sw["swing_delta"] > 0.055,
      f"delta {sw['swing_delta']}")

# Flam: same pattern, kicks 30ms late relative to the grid.
late_bands = synth_bands(BPM)
late = prep_signature(rhythm_signature(
    late_bands, [{**grid_for(BPM, 120.0)[0], "first_beat_s": -0.03}], 0,
    fps=FPS, latency_s=0.0))
fl = rhythm_terms(sig4, late, 1.0, period)
check("30ms-late hits read as flam risk",
      fl["flam_ms"] is not None and 15.0 <= fl["flam_ms"] <= 60.0,
      f"flam {fl['flam_ms']}")

print("== tempo multiple ==")
check("double-time read recovered", tempo_mult_for(170.0, 85.0, 1.0) == 2.0)
check("half-time read recovered", tempo_mult_for(85.0, 170.0, 1.0) == 0.5)
check("plain stretch stays 1:1", tempo_mult_for(128.0, 124.0, 1.032) == 1.0)
half = rhythm_terms(sig4, sig4, 2.0, period)
check("4x4 vs itself at 2x still locks kicks",
      half["kick_agreement"] > 0.55, f"agr {half['kick_agreement']}")

print("== seam entry point + chips ==")
a = SimpleNamespace(rhythm_sig=sig4, bpm=BPM, bpm_conf=0.9)
b = SimpleNamespace(rhythm_sig=offbeat, bpm=BPM, bpm_conf=0.4)
rt = seam_rhythm(a, b, rate=1.0)
check("seam_rhythm computes", rt is not None)
check("conf = weaker grid", rt["conf"] == 0.4)
chips = seam_chips({"rate": 1.0}, {"rhythm": rt, "mult": rt["mult"]})
check("kick clash chip present, '?'-marked (shaky grid)",
      any(c == "kick clash?" for c in chips), str(chips))
check("no signature -> no terms",
      seam_rhythm(SimpleNamespace(rhythm_sig=None, bpm=120), b) is None)
clean = seam_chips({"rate": 1.01},
                   {"rhythm": rhythm_terms(sig4, sig4, 1.0, period),
                    "mult": 1.0})
check("clean seam -> no chips", clean == [], str(clean))
big = seam_chips({"rate": 1.055}, {"mult": 0.5})
check("stretch + half-time chips (no signature needed)",
      "half-time" in big and any(c.startswith("stretch") for c in big),
      str(big))

print("== v2: regions ==")
# A track whose INTRO (first 60s) is offbeat-kick but whose body is
# four-on-floor: region view must expose the difference.
mixed = synth_bands(BPM, dur_s=180.0)
intro = synth_bands(BPM, dur_s=60.0, kick_16ths=(2, 6, 10, 14))
mixed[:int(60 * FPS)] = intro
msig = prep_signature(rhythm_signature(
    mixed, grid_for(BPM, 180.0), 0, fps=FPS, latency_s=0.0,
    mix_in_s=5.0, mix_out_s=170.0))
check("region keys stored", msig.get("in_low") is not None
      and msig.get("out_low") is not None)
vin = region_view(msig, "in")
vout = region_view(msig, "out")
check("in-region shows the offbeat intro",
      max(vin["low"][i] for i in on_steps) < 0.5,
      f"max on-step {max(vin['low'][i] for i in on_steps):.2f}")
check("out-region shows the 4x4 body",
      min(vout["low"][i] for i in on_steps) > 0.5,
      f"min on-step {min(vout['low'][i] for i in on_steps):.2f}")
check("v1 sig without regions falls back whole-track",
      region_view(sig4, "in") is sig4)
a4 = SimpleNamespace(rhythm_sig=sig4, bpm=BPM, bpm_conf=0.9)
bm = SimpleNamespace(rhythm_sig=msig, bpm=BPM, bpm_conf=0.9)
rt_reg = seam_rhythm(a4, bm, 1.0)
check("seam compares out vs in regions", rt_reg["regions"] == "full/in"
      and rt_reg["kick_agreement"] < 0.4,
      f"regions {rt_reg['regions']} agr {rt_reg['kick_agreement']}")

print("== v2: meter ==")
check("4/4 tracks read meter 4", sig4.get("meter") == 4)


def synth_waltz(bpm=120.0, dur_s=120.0):
    """3/4: kick on beat 1 of each 3-beat bar, snare on 2 and 3."""
    T = int(dur_s * FPS)
    rng = np.random.RandomState(3)
    bands = 0.05 + 0.01 * rng.rand(T, 32)
    period = 60.0 / bpm
    t = 0.0
    while t < dur_s - 1.0:
        for beat in range(3):
            f = int((t + beat * period) * FPS)
            if f + 2 >= T:
                break
            if beat == 0:
                bands[f:f + 2, 0:6] += 6.0
            else:
                bands[f:f + 2, 6:16] += 4.0
        t += 3 * period
    return bands


waltz = prep_signature(rhythm_signature(
    synth_waltz(BPM), grid_for(BPM, 120.0), 0, fps=FPS, latency_s=0.0))
check("waltz reads meter 3", waltz.get("meter") == 3,
      f"meter {waltz.get('meter')} conf {waltz.get('meter_conf')}")
mrt = rhythm_terms(sig4, waltz, 1.0, period)
check("3/4 vs 4/4 flagged + crushed", mrt["meter_clash"]
      and mrt["score"] < 0.35, f"score {mrt['score']}")
mchips = seam_chips({"rate": 1.0}, {"rhythm": mrt, "mult": 1.0})
check("meter chip present", any("/4 vs" in c for c in mchips), str(mchips))

print("== v2: plan steering ==")
from lib.dj.brain import Brain


class _Theme:
    pass


def ghost(sig, tid, title):
    row = {"id": tid, "path": f"{title}.mp3", "title": title, "artist": "x",
           "duration_s": 300.0, "bpm": BPM, "bpm_conf": 0.9,
           "downbeat_offset": 0, "downbeat_conf": 0.8, "camelot": "8A",
           "beat_grid": grid_for(BPM, 300.0), "loudness_gain_db": 0.0,
           "kick_offset_s": 0.0, "rhythm": None}
    from lib.dj.brain import TrackInfo
    secs = [{"kind": "groove", "start_s": 0.0, "end_s": 300.0,
             "start_beat": 0, "end_beat": 600, "energy": 0.8,
             "bass_share": 0.4, "mid_share": 0.4, "high_share": 0.2,
             "rhythm_density": 2.0, "repetitiveness": 0.7,
             "busyness": 0.4, "vocalness": 0.0, "boundary_strength": 0.5}]
    mps = [{"kind": "in", "time_s": 30.0, "score": 0.5, "style_hint": "blend"},
           {"kind": "out", "time_s": 250.0, "score": 0.5,
            "style_hint": "blend"}]
    t = TrackInfo(row, secs, [], mps)
    t.rhythm_sig = sig
    return t


from lib.dj.themes import get_theme
ta = ghost(sig4, 1, "fourfloor")
tb = ghost(offbeat, 2, "offbeatkick")   # kick clash vs ta
brain = Brain([ta, tb], get_theme("groove"), seed=4)
styles = {}
for s in range(40):
    brain.rng.seed(s)
    brain.recent_styles = []
    plan = brain.plan_transition(ta, tb, {"rate": 1.0, "eff_bpm": BPM,
                                          "pair": None, "pitch_st": 0})
    styles[plan["style"]] = styles.get(plan["style"], 0) + 1
check("plan stamps rhythm terms", plan.get("rhythm") is not None
      and plan["rhythm"]["kick_agreement"] < 0.4)
open_low = styles.get("long_blend", 0) + styles.get("bassline_layer", 0)
check("kick clash avoids open-low styles",
      open_low <= 0.2 * sum(styles.values()), str(styles))
tw = ghost(waltz, 3, "waltz")
brain2 = Brain([ta, tw], get_theme("groove"), seed=4)
p2 = brain2.plan_transition(ta, tw, {"rate": 1.0, "eff_bpm": BPM,
                                     "pair": None, "pitch_st": 0})
check("meter clash forces long_fade", p2["style"] == "long_fade",
      p2["style"])

print(f"\n{CHECKS['pass']} passed, {CHECKS['fail']} failed")
sys.exit(0 if CHECKS["fail"] == 0 else 1)
