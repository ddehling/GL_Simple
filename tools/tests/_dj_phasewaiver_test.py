"""GATE: the phase-profile waiver is the correction's own lookup.

Born 2026-08-16, from a live seam that measured clean and sounded
terrible: stem_drum_swap played at bpm_conf 0.69/0.66 because the
grid-confidence wall stood down on `_local_ok` - a region-DEFAULT
phase_offset lookup - while the kick-true anchor correction (region
'out'/'in') found nothing and applied 0.0 (the armed log's
phase_a_ms: 0.0). Grids "locked" to 5ms; the rendered kicks flammed
125ms median. The gates stood down on evidence the seam never received.

The invariant this gate holds: for any phase-file format (profile,
legacy-with-positions, legacy label-only) and any anchor, the lookup
that WAIVES a safety gate must return None exactly where the lookup
that CORRECTS the anchors returns None. No sleep()s: all cases live in
one synthetic beat_power.json so the module's mtime cache loads once.

Usage:
    python tools/tests/_dj_phasewaiver_test.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

failures = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def main():
    doc = {"scores": {
        # 1: profile format - trusted buckets around 100s only.
        "1": {"phase": {"prof": [
            {"at_s": 90.0, "ms": 20.0, "iqr": 4.0, "n": 30},
            {"at_s": 110.0, "ms": 24.0, "iqr": 4.0, "n": 30},
        ]}},
        # 2: legacy records WITH positions - one trusted rec at 10s.
        "2": {"phase": {
            "in": {"at_s": 10.0, "ms": 15.0, "iqr": 4.0, "n": 30},
            "out": {"at_s": 200.0, "ms": -30.0, "iqr": 90.0, "n": 5},
        }},
        # 3: legacy label-only (no positions) - trusted 'mid',
        #    UNTRUSTED 'out'. The live-failure shape: a track-body
        #    measurement exists, the exit region's does not.
        "3": {"phase": {
            "mid": {"ms": 31.0, "iqr": 5.0, "n": 40},
            "out": {"ms": -12.0, "iqr": 120.0, "n": 4},
        }},
    }}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "beat_power.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.environ["DJ_BEATPOWER_PATH"] = p
        from lib.dj import beatpower as bp

        print("profile format:")
        check("prof in reach", bp.phase_offset(1, region="out",
                                               at_s=100.0) is not None,
              "trusted bucket 10s away -> value")
        check("prof beyond reach", bp.phase_offset(1, region="out",
                                                   at_s=200.0) is None,
              "nearest trusted bucket 90s away -> None")

        print("legacy records with positions:")
        check("legacy in reach", bp.phase_offset(2, region="out",
                                                 at_s=15.0) is not None,
              "trusted rec 5s away -> value")
        check("legacy beyond reach", bp.phase_offset(2, region="out",
                                                     at_s=100.0) is None,
              "nearest trusted rec 90s away -> None (reach rule)")

        print("legacy label-only (the live-failure shape):")
        # The CORRECTION asks region='out' at the exit anchor. It must
        # get None (the out record is untrusted)...
        out_corr = bp.phase_offset(3, region="out", at_s=299.7)
        check("correction declines", out_corr is None,
              f"region='out' -> {out_corr}")
        # ...and the WAIVER must agree with it. The pre-2026-08-16
        # region-default call returned the trusted 'mid' record here,
        # standing gates down on evidence the seam never received.
        waive_old = bp.phase_offset(3, at_s=299.7)          # region-default
        waive_new = bp.phase_offset(3, region="out", at_s=299.7)
        check("waiver==correction", (waive_new is None) == (out_corr is None),
              f"region='out' waiver {waive_new} vs correction {out_corr}")
        if waive_old is not None:
            print("  (note: region-default lookup still returns "
                  f"{waive_old} - callers must pass region explicitly, "
                  "which brain/gateprobe now do)")

        # gateprobe's mirror must run the region-specific form.
        import inspect
        from lib.dj import gateprobe
        src = inspect.getsource(gateprobe.local_phase_known)
        check("gateprobe mirrors", 'region="out"' in src
              and 'region="in"' in src, "local_phase_known region-specific")
        from lib.dj import brain as _brain
        bsrc = inspect.getsource(_brain.Brain)
        check("brain waiver region-specific",
              bsrc.count('phase_offset(cur.id, region="out"') >= 1
              and bsrc.count('phase_offset(cand.id, region="in"') >= 1,
              "_local_ok asks the correction's question")

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("phase-waiver gate: all clear")


if __name__ == "__main__":
    main()
