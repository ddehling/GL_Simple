"""Auto-tune a recreation against its score: coordinate descent over
each section's levers, rendering only that section and measuring its
local score against the original's bars, so a 2-minute song tunes in a
couple of minutes.

    tuned, report = tune(res, script, rounds=2, progress=cb)

Per section, per round, every lever below is tried a step up and a step
down (and each optional layer toggled); a move is kept when the section's
mean local score improves. Levers: energy, density, brightness, swing,
the low-pass / high-pass lanes, and the layer set. Style, key, tempo,
chords and material are left alone - they are decisions, not levels.

The report lists every accepted move with its gain, so the operator can
see what the tuner thought the song wanted."""
from __future__ import annotations

import copy

import numpy as np

from lib.gen import RATE

STEPS = {"energy": 0.15, "density": 0.2, "brightness": 0.2, "swing": 0.05}
BOUNDS = {"energy": (0.05, 1.0), "density": (0.4, 1.5), "brightness": (0.4, 1.6), "swing": (0.0, 0.33)}
LP_CHOICES = (20000.0, 8000.0, 4000.0, 2000.0)
HP_CHOICES = (20.0, 120.0, 300.0)
TOGGLE_LAYERS = ("lead", "arp", "keys", "hat", "shaker", "ohat", "ride", "perc", "pad")


MAX_TUNE_BARS = 16      # a long section is judged on its first 16 bars (4x faster, same levers)


def _render_section(script, idx, seconds_pad=1.0):
    """Render only section idx (its first MAX_TUNE_BARS bars) and return per-bar features on the grid."""
    from lib.gen import script as S
    from lib.gen.analysis import ingest as I
    sub = copy.deepcopy(script)
    sub["sections"] = [copy.deepcopy(script["sections"][idx])]
    sub["sections"][0]["bars"] = min(int(sub["sections"][0]["bars"]), MAX_TUNE_BARS)
    sub["end"] = False
    sub["vocals"] = []
    audio, _ = S.render(sub, seconds=S.total_seconds(sub, sub["bpm"]) + seconds_pad)
    return I.features_on_grid(audio.mean(axis=1).astype(np.float32), sub["bpm"], 0.0)


def _section_bars(script, idx):
    bar = sum(int(e["bars"]) for e in script["sections"][:idx])
    return bar, bar + min(int(script["sections"][idx]["bars"]), MAX_TUNE_BARS)


def _section_score(orig_feats, sec_feats, b0, b1):
    from lib.gen.analysis import score as SC
    o = orig_feats[b0:b1]
    if not o or not sec_feats:
        return 0.0
    rep = SC.compare(o, sec_feats, align=False)
    return rep["mean_local"]


def tune(res, script, rounds: int = 2, progress=None, sections=None):
    """res: an ingest result (needs res["features"]); script: the SongScript
    to tune (not modified). Returns (tuned script, report)."""
    from lib.gen import script as S
    sc = S.normalize(script)
    feats = res["features"]
    report = {"moves": [], "before": {}, "after": {}}
    idxs = list(sections) if sections is not None else list(range(len(sc["sections"])))
    total = max(1, rounds * len(idxs))
    done = 0
    for rnd in range(rounds):
        for idx in idxs:
            b0, b1 = _section_bars(sc, idx)
            if b0 >= len(feats):
                continue
            base = _section_score(feats, _render_section(sc, idx), b0, b1)
            report["before"].setdefault(idx, base)
            e = sc["sections"][idx]
            # numeric levers
            for lever, step in STEPS.items():
                cur = float(e.get(lever, {"energy": 0.6, "density": 1.0, "brightness": 1.0, "swing": 0.08}[lever]))
                best_val, best = cur, base
                for cand in (cur + step, cur - step):
                    lo, hi = BOUNDS[lever]
                    cand = float(np.clip(cand, lo, hi))
                    if abs(cand - cur) < 1e-6:
                        continue
                    trial = copy.deepcopy(sc)
                    trial["sections"][idx][lever] = round(cand, 3)
                    s = _section_score(feats, _render_section(trial, idx), b0, b1)
                    if s > best + 0.3:
                        best_val, best = cand, s
                if best_val != cur:
                    report["moves"].append({"round": rnd, "section": idx, "lever": lever, "from": round(cur, 3), "to": round(best_val, 3),
                                            "gain": round(best - base, 1)})
                    e[lever] = round(best_val, 3)
                    base = best
            # lanes
            lanes = dict(e.get("lanes") or {})
            for lane, choices in (("lp", LP_CHOICES), ("hp", HP_CHOICES)):
                cur = float(lanes.get(lane, choices[0]))
                best_val, best = cur, base
                for cand in choices:
                    if abs(cand - cur) < 1e-6:
                        continue
                    trial = copy.deepcopy(sc)
                    tl = dict(trial["sections"][idx].get("lanes") or {})
                    tl[lane] = cand
                    trial["sections"][idx]["lanes"] = tl
                    s = _section_score(feats, _render_section(trial, idx), b0, b1)
                    if s > best + 0.3:
                        best_val, best = cand, s
                if best_val != cur:
                    lanes[lane] = best_val
                    e["lanes"] = lanes
                    report["moves"].append({"round": rnd, "section": idx, "lever": f"lane:{lane}", "from": cur, "to": best_val, "gain": round(best - base, 1)})
                    base = best
            # layers: toggle one at a time
            if e.get("layers") is not None:
                for slot in TOGGLE_LAYERS:
                    cur = list(e["layers"])
                    cand = [x for x in cur if x != slot] if slot in cur else sorted(cur + [slot])
                    trial = copy.deepcopy(sc)
                    trial["sections"][idx]["layers"] = cand
                    s = _section_score(feats, _render_section(trial, idx), b0, b1)
                    if s > base + 0.5:
                        report["moves"].append({"round": rnd, "section": idx, "lever": f"layer:{slot}", "from": slot in cur, "to": slot in cand,
                                                "gain": round(s - base, 1)})
                        e["layers"] = cand
                        base = s
            report["after"][idx] = base
            done += 1
            if progress:
                progress(done / total, f"tuned section {idx} ({e['section']}): {report['before'][idx]:.1f} -> {base:.1f}")
    report["gain"] = round(float(np.mean([report["after"][i] - report["before"][i] for i in report["after"]])) if report["after"] else 0.0, 1)
    return sc, report
