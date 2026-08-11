#!/usr/bin/env python3
"""Offline check of the per-receiver correction knobs (gain + gamma).

The knobs: a receiver in ``project.yaml`` may declare ``gamma:`` to
correct itself independently of the web control panel's Gamma slider
(0, anything under 0.01, means "follow the slider"; a larger value is
that box's exponent), and ``gain:`` to sit brighter or quieter than the
rest of the piece (1.0 = no emphasis, 0 mutes the box). Both apply to
that box's pixels only.

This gate runs the real resolution path — ``core.strip`` builds the
strip pixel indices, the metadata-atlas construction is mirrored from
``Stories_OGL._publish_strip_metadata``, and ``core.correction`` resolves
and applies — with no GL, no network, and no hardware. Every applied
output is compared against a naive per-pixel reference (plain ``pow``
plus a multiply) so a bug in table construction, region resolution or
the cache surfaces as a mismatch instead of cancelling out.

What it verifies
  1. Value parsing — for gamma, 0/missing/negative/NaN/junk all read as
     System and 0.01+ reads as an override; for gain, missing/negative/
     NaN/junk read as neutral while 0 legitimately mutes.
  2. Region geometry — a correction lands on exactly the canvas pixels
     its own receiver drives, and on nothing else. Checked against every
     real project, for row-strip and column-strip layouts alike. This is
     what a per-ROW gain vector got wrong on Fan, where a box owns
     columns rather than rows. Contiguous boxes must reduce to slices
     (cheap to write); irregular ones must keep a boolean mask.
  3. Applied output — matches the reference for gamma alone, gain alone,
     both on one box, and a muted box; gammas never compound; an
     overridden box ignores the slider while a gain-only box follows it.
  4. Uncorrected groups are bit-identical to the old scalar pow path.
  5. Table cache — tables are built once per slider value and rebuilt
     only when it moves, and each table folds gamma and gain together.

Usage:
    python tools/tests/_receiver_correction_test.py            # every project
    python tools/tests/_receiver_correction_test.py --project fan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.correction import (  # noqa: E402
    MIN_OVERRIDE, NEUTRAL_GAIN, SYSTEM, apply_correction,
    build_correction_plans, parse_gain, parse_gamma,
)
from core.strip import strips_from_yaml_list  # noqa: E402

FAILURES: list[str] = []


def check(cond: bool, label: str) -> bool:
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}")
        FAILURES.append(label)
    return cond


# ---------------------------------------------------------------------------
# Project fixtures
# ---------------------------------------------------------------------------

def load_project_yaml(project_id: str) -> dict:
    path = ROOT / "projects" / project_id / "project.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_atlases(raw: dict) -> dict:
    """Mirror of the receiver_idx half of _publish_strip_metadata.

    Deliberately re-derived here rather than imported from Stories_OGL
    (which needs GL + audio at import time); if the two ever diverge the
    region geometry check below is what notices.
    """
    groups = raw.get("groups") or []
    dims = {str(g["id"]): (int(g["height"]), int(g["width"])) for g in groups}
    atlases = {
        gid: {"receiver_idx": np.full((h, w), -1, dtype=np.int32)}
        for gid, (h, w) in dims.items()
    }
    for rx_idx, rx in enumerate(raw.get("receivers") or []):
        if not isinstance(rx, dict):
            continue
        for s in strips_from_yaml_list(rx.get("strips", []) or []):
            atlas = atlases.get(s.group_id)
            if atlas is None:
                continue
            idx = np.asarray(s.pixel_indices, dtype=np.int32)
            if idx.size == 0:
                continue
            rows, cols = idx[:, 0], idx[:, 1]
            h, w = atlas["receiver_idx"].shape
            ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            atlas["receiver_idx"][rows[ok], cols[ok]] = rx_idx
    return atlases


def random_frames(atlases: dict, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    return {
        gid: rng.integers(0, 256, size=(*a["receiver_idx"].shape, 3),
                          dtype=np.uint8)
        for gid, a in atlases.items()
    }


# ---------------------------------------------------------------------------
# 1. Value parsing
# ---------------------------------------------------------------------------

def test_parse():
    print("\n[1] value parsing")
    system_cases = [0, 0.0, "0", None, "", "banana", -1.0, 0.009,
                    float("nan"), float("-inf"), [1, 2]]
    ok = all(parse_gamma(v) == SYSTEM for v in system_cases)
    check(ok, "gamma: 0 / missing / negative / NaN / junk all read as System")

    override_cases = {MIN_OVERRIDE: MIN_OVERRIDE, 1.0: 1.0, 2.4: 2.4,
                      "2.6": 2.6, 3: 3.0}
    ok = all(parse_gamma(k) == v for k, v in override_cases.items())
    check(ok, "gamma: 0.01 and up read as an override (ints and strings too)")

    neutral_cases = [None, "", "banana", -0.5, float("nan"), float("inf"),
                     {1: 2}]
    ok = all(parse_gain(v) == NEUTRAL_GAIN for v in neutral_cases)
    check(ok, "gain: missing / negative / NaN / junk all read as neutral 1.0")

    gain_cases = {0: 0.0, 0.5: 0.5, "1.35": 1.35, 2: 2.0}
    ok = all(parse_gain(k) == v for k, v in gain_cases.items())
    check(ok, "gain: 0 is a legitimate mute, not a typo; values pass through")


# ---------------------------------------------------------------------------
# Independent reference: the naive per-pixel implementation, no tables.
# ---------------------------------------------------------------------------

def reference(frame, owner, receivers, system_gamma):
    out = np.power(frame / 255.0, system_gamma) * 255.0
    for rx_idx, rx in enumerate(receivers):
        gamma = parse_gamma(rx.get("gamma", SYSTEM))
        gain = parse_gain(rx.get("gain", NEUTRAL_GAIN))
        if gamma == SYSTEM and gain == NEUTRAL_GAIN:
            continue
        sel = (owner == rx_idx)
        if not sel.any():
            continue
        g = system_gamma if gamma == SYSTEM else gamma
        out[sel] = np.power(frame[sel] / 255.0, g) * 255.0 * gain
    return out


def apply_all(plans, atlases, frames, receivers, system_gamma):
    """Run the real path over every group; return (ok, corrected_frames)."""
    ok = True
    outs = {}
    for gid, frame in frames.items():
        out = apply_correction(frame, system_gamma, plans.get(gid))
        want = reference(frame, atlases[gid]["receiver_idx"], receivers,
                         system_gamma)
        ok &= np.allclose(out, want)
        outs[gid] = out
    return ok, outs


# ---------------------------------------------------------------------------
# 2. Region geometry + applied output, per project
# ---------------------------------------------------------------------------

def test_project(project_id: str):
    print(f"\n[2] {project_id}: region geometry + applied output")
    raw = load_project_yaml(project_id)
    receivers = raw.get("receivers") or []
    atlases = build_atlases(raw)
    frames = random_frames(atlases)

    check(bool(receivers) and bool(atlases),
          f"{project_id}: {len(receivers)} receivers, "
          f"{len(atlases)} group canvas(es) loaded")
    if not receivers:
        return

    # -- as shipped ------------------------------------------------------
    # Whatever the project currently declares — no overrides at all (Fan),
    # gain emphasis (WoL's central sculpture), or a gamma per box (WoL's
    # ring) — must resolve to output that matches the reference. This is
    # deliberately not an assertion about WHICH boxes are corrected: that
    # is a tuning decision the operator makes in the layout editor, and a
    # gate that pins it would fail every time the rig is tuned.
    shipped = build_correction_plans(receivers, atlases)
    on_slider = sum(1 for rx in receivers
                    if parse_gamma(rx.get("gamma", SYSTEM)) == SYSTEM)
    print(f"      as shipped: {len(receivers) - on_slider} of "
          f"{len(receivers)} boxes override gamma, {on_slider} on the slider")

    emphasised = sum(px for plan in shipped.values()
                     for px, _, gain in plan.summary() if gain != NEUTRAL_GAIN)
    ok, _ = apply_all(shipped, atlases, frames, receivers, 2.0)
    check(ok, f"{project_id}: shipped correction matches the naive per-pixel "
              f"reference ({emphasised} px emphasised by gain)")

    # Groups nobody corrects must be bit-identical to the old scalar path.
    same = True
    for gid, frame in frames.items():
        if gid in shipped:
            continue
        same &= np.array_equal(apply_correction(frame, 2.0, None),
                               np.power(frame / 255.0, 2.0) * 255.0)
    check(same, f"{project_id}: uncorrected groups are bit-identical to the "
                f"old scalar pow path")

    # -- both knobs, on two different boxes -------------------------------
    target = len(receivers) - 1
    patched = [dict(rx) for rx in receivers]
    patched[target]["gamma"] = 2.6          # override...
    patched[target]["gain"] = 0.8           # ...and emphasis, same box
    if len(receivers) >= 2:
        patched[0]["gamma"] = 1.6
        patched[0]["gain"] = 0.0            # muted box
    plans = build_correction_plans(patched, atlases)

    ok, outs = apply_all(plans, atlases, frames, patched, 2.0)
    check(ok, f"{project_id}: gamma + gain on the same box, and a second box "
              f"muted, both match the reference")

    # Region geometry: rebuild the per-pixel (gamma, gain) map from the
    # regions and compare it against the map the YAML implies. Value-based
    # rather than identity-based, so it holds however many boxes are
    # corrected and whatever they are set to.
    covered = 0
    clean = True
    for gid, plan in plans.items():
        owner = atlases[gid]["receiver_idx"]
        want_gamma = np.zeros(owner.shape)
        want_gain = np.ones(owner.shape)
        for rx_idx, rx in enumerate(patched):
            sel = (owner == rx_idx)
            if sel.any():
                want_gamma[sel] = parse_gamma(rx.get("gamma", SYSTEM))
                want_gain[sel] = parse_gain(rx.get("gain", NEUTRAL_GAIN))
        got_gamma = np.zeros(owner.shape)
        got_gain = np.ones(owner.shape)
        for region, gamma, gain, px in plan.regions:
            got_gamma[region] = gamma
            got_gain[region] = gain
            covered += px
        clean &= np.array_equal(got_gamma, want_gamma)
        clean &= np.array_equal(got_gain, want_gain)
    check(clean and covered > 0,
          f"{project_id}: every region carries its own receiver's exact "
          f"gamma and gain, on exactly that receiver's pixels ({covered} px "
          f"corrected, no bleed)")

    # Contiguous boxes must resolve to slices, not boolean masks — that is
    # what keeps the per-frame write cheap. Both shipped projects give each
    # box a solid run of rows (WoL) or columns (Fan).
    sliced = all(isinstance(region, tuple)
                 for plan in plans.values()
                 for region, _, _, _ in plan.regions)
    check(sliced, f"{project_id}: contiguous boxes resolve to slice regions "
                  f"rather than boolean masks")

    # -- no compounding ---------------------------------------------------
    isolated = True
    for gid, frame in frames.items():
        sel = (atlases[gid]["receiver_idx"] == target)
        if not sel.any():
            continue
        # If the override were layered on the system output instead of
        # replacing it, the values would be (x^2.0)^2.6 = x^5.2.
        stacked = np.power(np.power(frame[sel] / 255.0, 2.0), 2.6) * 255.0 * 0.8
        lit = frame[sel] > 8       # near-black is identical either way
        if lit.any():
            isolated &= not np.allclose(outs[gid][sel][lit], stacked[lit])
    check(isolated, f"{project_id}: exponents don't compound (an override "
                    f"replaces the system gamma, it doesn't stack on it)")

    # -- the system slider must not move an overridden box ----------------
    stable = True
    for gid, frame in frames.items():
        sel = (atlases[gid]["receiver_idx"] == target)
        if not sel.any():
            continue
        a = apply_correction(frame, 1.4, plans.get(gid))[sel]
        b = apply_correction(frame, 2.9, plans.get(gid))[sel]
        stable &= np.allclose(a, b)
    check(stable, f"{project_id}: dragging the system slider 1.4 -> 2.9 "
                  f"leaves the fully-overridden box unchanged")

    # A gain-only box, by contrast, MUST follow the slider (its table is
    # the system table times its gain) — that is the whole meaning of
    # gamma 0.
    if emphasised:
        follows = False
        for gid, frame in frames.items():
            owner = atlases[gid]["receiver_idx"]
            for rx_idx, rx in enumerate(receivers):
                if parse_gain(rx.get("gain", NEUTRAL_GAIN)) == NEUTRAL_GAIN:
                    continue
                sel = (owner == rx_idx)
                if not sel.any():
                    continue
                a = apply_correction(frame, 1.4, shipped.get(gid))[sel]
                b = apply_correction(frame, 2.9, shipped.get(gid))[sel]
                follows |= not np.allclose(a, b)
        check(follows, f"{project_id}: a gain-only box still tracks the "
                       f"system slider (gamma 0 means System)")


# ---------------------------------------------------------------------------
# 3. The table cache: built once, rebuilt only when the slider moves
# ---------------------------------------------------------------------------

def test_table_cache():
    print("\n[3] lookup-table cache")
    owner = np.full((8, 8), -1, dtype=np.int32)
    owner[:, :4] = 0                       # box 0 owns the left half
    plan = build_correction_plans([{"gamma": 2.6, "gain": 1.2}],
                                  {"g": {"receiver_idx": owner}})["g"]
    frame = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    calls = []
    real_rebuild = plan._rebuild

    def counting_rebuild(system_gamma):
        calls.append(system_gamma)
        real_rebuild(system_gamma)
    plan._rebuild = counting_rebuild

    for _ in range(10):
        plan.apply(frame, 2.0)
    check(calls == [2.0], "10 frames at one slider value build the tables once")

    plan.apply(frame, 2.05)
    plan.apply(frame, 2.05)
    check(calls == [2.0, 2.05],
          "moving the slider rebuilds once, then caches again")

    check(len(plan._tables) == len(plan.regions) and plan._base.shape == (256,),
          "one 256-entry table per region, plus the base table")

    ramp = np.arange(256) / 255.0
    check(np.allclose(plan._tables[0], np.power(ramp, 2.6) * 255.0 * 1.2),
          "a region's table folds its gamma and gain into one lookup")

    # A gain-only region tracks the slider through the base table.
    gain_only = build_correction_plans([{"gain": 1.5}],
                                       {"g": {"receiver_idx": owner}})["g"]
    gain_only.apply(frame, 2.2)
    check(np.allclose(gain_only._tables[0], np.power(ramp, 2.2) * 255.0 * 1.5),
          "a gain-only region's table is the system table times its gain")


# ---------------------------------------------------------------------------
# 4. Defensive cases
# ---------------------------------------------------------------------------

def test_defensive():
    print("\n[4] defensive cases")

    owner = np.full((4, 4), -1, dtype=np.int32)
    owner[:, :2] = 0
    plan = build_correction_plans([{"gamma": 2.6}],
                                  {"g": {"receiver_idx": owner}})["g"]

    big = np.full((8, 8, 3), 200, dtype=np.uint8)
    out = plan.apply(big, 2.0)
    check(np.allclose(out, np.power(big / 255.0, 2.0) * 255.0),
          "canvas resized under a live plan: the stale region is skipped and "
          "every pixel keeps the system correction, instead of the correction "
          "landing in the wrong place")

    # The pipeline hands us a non-contiguous RGBA slice when the canvas
    # carries an alpha channel — table indexing has to survive that.
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[..., :3] = 200
    out = plan.apply(rgba[:, :, :3], 2.0)
    sel = np.zeros((4, 4), dtype=bool)
    sel[:, :2] = True
    ok = (np.allclose(out[sel], np.power(200 / 255.0, 2.6) * 255.0)
          and np.allclose(out[~sel], np.power(200 / 255.0, 2.0) * 255.0))
    check(ok, "RGBA canvas: correction applies through the pipeline's "
              "non-contiguous [:, :, :3] view")

    src = np.full((4, 4, 3), 123, dtype=np.uint8)
    before = src.copy()
    plan.apply(src, 2.0)
    check(np.array_equal(src, before), "the input frame is left unmodified")

    # An irregular region (a polyline strip that skips pixels) has to keep
    # its boolean mask rather than being widened to a bounding box.
    ragged = np.full((4, 4), -1, dtype=np.int32)
    ragged[0, 0] = 0
    ragged[2, 2] = 0
    rp = build_correction_plans([{"gamma": 2.6}],
                                {"g": {"receiver_idx": ragged}})["g"]
    region = rp.regions[0][0]
    check(not isinstance(region, tuple) and int(np.count_nonzero(region)) == 2,
          "an irregular box keeps a boolean mask (no bounding-box overreach)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="only this project id")
    args = ap.parse_args()

    if args.project:
        ids = [args.project]
    else:
        ids = sorted(p.parent.name for p in
                     (ROOT / "projects").glob("*/project.yaml"))

    print("=" * 70)
    print("PER-RECEIVER OUTPUT CORRECTION (gain + gamma)")
    print("=" * 70)
    test_parse()
    for pid in ids:
        test_project(pid)
    test_table_cache()
    test_defensive()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
