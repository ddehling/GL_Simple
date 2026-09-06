"""Gate for the generator -> visuals coupling (lib/gen/vis.py +
GenSystem.live_beat). Offline: plain state dicts, no renderer.

  1. INACTIVE: every key passes through untouched.
  2. ENERGY: smoothed ground truth replaces the DSP estimate (~0.8 s).
  3. BUILD: build_level ramps to 1 through the last 8 s before a known drop.
  4. DROP: a new gen_drop_t stamp fires drop once and decays drop_decay.
  5. BEAT: onsets from the composer's phase pulse the room when the phrase
     has a kick; a break (no kick) keeps phases gliding but never flashes.
  6. LIVE_BEAT from a real GenSystem: bpm exact, phase in [0,1), drive 1 in
     a kick section and 0 when the kick is muted.

Usage: python tools/tests/_gen_vis_test.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.system import GenSystem                      # noqa: E402
from lib.gen.vis import GenVisualCoupler                  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def base():
    return {"audio_energy": 0.42, "build_level": 0.1, "drop": False, "drop_decay": 0.0, "beat": False, "beat_decay": 0.0,
            "beat_phase": 0.3, "bar_phase": 0.1, "phrase_phase": 0.05, "beat_confidence": 0.2, "beat_intensity": 0.0,
            "bass_punch": 0.0, "audio_punch": 0.0, "bpm": 100.0}


def main():
    print("== inactive")
    v = GenVisualCoupler(); s = base(); s.update({"gen_active": False, "gen_energy": 0.9, "gen_next_drop_eta": 1.0, "gen_drop_t": 5.0})
    v.apply(s, 0.025, {"bpm": 124, "phase": 0.9, "bar_phase": 0.5, "phrase_phase": 0.5, "bass_share": 0.5, "drive": 1.0})
    check(s["audio_energy"] == 0.42 and s["build_level"] == 0.1 and not s["drop"] and s["bpm"] == 100.0, "no-op while inactive")

    print("== energy")
    v = GenVisualCoupler(); s = base(); s.update({"gen_active": True, "gen_energy": 0.9})
    v.apply(s, 0.025, None); e1 = s["audio_energy"]
    for _ in range(80):
        v.apply(s, 0.025, None)
    check(e1 == 0.9 and abs(s["audio_energy"] - 0.9) < 1e-6, "ground-truth energy replaces the DSP value")
    s["gen_energy"] = 0.2; v.apply(s, 0.025, None)
    check(0.85 < s["audio_energy"] < 0.9, f"smoothed, not stepped ({s['audio_energy']:.3f})")

    print("== build + drop")
    v = GenVisualCoupler(); s = base(); s.update({"gen_active": True, "gen_energy": 0.7, "gen_next_drop_eta": 12.0})
    v.apply(s, 0.025, None); b12 = s["build_level"]
    s["gen_next_drop_eta"] = 4.0; s["build_level"] = 0.0; v.apply(s, 0.025, None); b4 = s["build_level"]
    s["gen_next_drop_eta"] = 0.2; s["build_level"] = 0.0; v.apply(s, 0.025, None); b0 = s["build_level"]
    check(b12 == 0.1 and abs(b4 - 0.5) < 1e-6 and b0 > 0.97, f"build ramps through the last 8 s: {b12}, {b4:.2f}, {b0:.2f}")
    s["gen_drop_t"] = 1000.0; s["drop"] = False; v.apply(s, 0.025, None)
    check(s["drop"] is True and s["drop_decay"] >= 0.99, "drop fires on a new stamp")
    s["drop"] = False; s["drop_decay"] = 0.0; v.apply(s, 0.025, None)
    check(s["drop"] is False and 0.9 < s["drop_decay"] < 1.0, "same stamp does not re-fire; decay runs")
    for _ in range(100):
        s["drop_decay"] = 0.0; v.apply(s, 0.025, None)
    check(s["drop_decay"] < 0.05, "drop decays away")

    print("== beat")
    v = GenVisualCoupler(); s = base(); s["gen_active"] = True
    def lb(ph, drive=1.0):
        return {"bpm": 124.0, "phase": ph, "bar_phase": 0.25, "phrase_phase": 0.5, "bass_share": 0.5, "drive": drive}
    v.apply(s, 0.025, lb(0.8)); s["beat"] = False; s["beat_intensity"] = 0.0
    v.apply(s, 0.025, lb(0.05))
    check(s["beat"] is True and s["beat_intensity"] > 0.9 and s["bpm"] == 124.0 and s["beat_phase"] == 0.05 and s["beat_confidence"] >= 0.99, "onset pulses with a kick")
    # a break: the kick's pulse envelope dies out, then onsets must not re-arm it
    for i in range(40):
        s["beat"] = False; s["beat_intensity"] = 0.0; v.apply(s, 0.025, lb((i * 0.13) % 1.0, drive=0.0))
    s["beat"] = False; s["beat_intensity"] = 0.0; v.apply(s, 0.025, lb(0.9, drive=0.0)); v.apply(s, 0.025, lb(0.05, drive=0.0))
    check(s["beat"] is False and s["beat_intensity"] < 0.05 and s["bar_phase"] == 0.25, "a break glides phases but does not flash")

    print("== live_beat from the system")
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=4, threaded=False, log_dir=tempfile.mkdtemp())
    g.start()
    def pump(sec):
        for _ in range(int(sec * RATE / 2048)):
            g.rack.read(2048); g.step()
    pump(20)
    b = g.live_beat()
    check(b is not None and abs(b["bpm"] - 124.0) < 1e-9 and 0.0 <= b["phase"] < 1.0 and 0.0 <= b["bar_phase"] < 1.0, f"live_beat {b}")
    check(b["drive"] == 1.0, "drive 1 with the kick playing")
    g.set_mute("kick", True); pump(9)
    check(g.live_beat()["drive"] < 1.0, "muting the kick lowers drive (no pulses on a resting kick)")
    ok = g.outstate_keys()
    check(ok["gen_active"] and "gen_next_drop_eta" in ok and "gen_drop_t" in ok, "outstate carries eta/drop for the coupler")
    g.stop()
    check(g.live_beat() is None, "None when idle")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
