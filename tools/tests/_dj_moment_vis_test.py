"""Gate + trace for the operator-MOMENT LIGHT choreography.

The rig isn't needed to validate what the room will do: the shaders see
only the reactive state keys, and the whole DJ->visuals mapping lives in
lib/dj/vis.DJVisualCoupler (extracted from Stories_OGL precisely so this
gate exercises the exact code that ships). This drives the real DJSystem
outstate (real submix render, real _do_moment) through the real coupler
per audio block and measures the choreography curves:

  1. Before the press the grid-pulse floor is beating the room.
  2. The build ramps to max INTO the landing (dj_next_drop_eta ramp).
  3. THE BREATH-HOLD: through the hole the build pins at 1.0 and the
     beat pulses are suppressed - the room holds its breath.
  4. The landing stamps drop at the hit, drop_decay from 1.0.
  5. HARD SLAM: an engineered landing decays ~3x slower than a natural
     musical drop (second run, playhead crossing a drop section).
  6. The published moment ETA counts down monotonically.

Usage:
    python tools/tests/_dj_moment_vis_test.py             # ALL PASS gate
    python tools/tests/_dj_moment_vis_test.py trace.png   # + curve plot
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _dj_moment_test as T                       # noqa: E402
from lib.dj.submix import RATE                    # noqa: E402
from lib.dj.vis import DJVisualCoupler            # noqa: E402

BLOCK = 2048
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


def vis_system():
    """The moment-test harness + what outstate_keys/live_beat read.
    Uses the nextdrop harness (the one surviving gesture): both the live
    track AND the queued next get the vis decorations, because outstate
    reads the INCOMING track after the landing's handover."""
    s = T.nextdrop_system("playing")
    for tr in (s.current, s.next_track):
        tr.bpm_conf = 0.9
        tr.grid = [{"start_s": 0.0, "end_s": T.DUR, "period_s": tr.period_s,
                    "first_beat_s": 0.0}]
        tr.row = {"downbeat_offset": 0}
        tr.sections = [dict(sec, rhythm_density=8.0, bass_share=0.5)
                       for sec in type(tr).sections]
    s._running = True
    s.arc_progress = lambda: 0.5
    s.arc_target = lambda: 0.5
    s.live_energy = lambda: 0.7
    return s


def run(seconds, fire_at=None, cue_to=None, flavor="nextdrop"):
    """Render + couple, returning per-block traces of the shader keys."""
    s = vis_system()
    if cue_to is not None:
        s.submix.post({"cmd": "cue", "deck": "a", "time_s": cue_to})
    coupler = DJVisualCoupler()
    tr = {k: [] for k in ("t", "audio", "build", "drop_decay", "drop",
                          "beat_decay", "beat_intensity", "hole", "eta")}
    dt = BLOCK / RATE
    for i in range(int(seconds * RATE / BLOCK)):
        now = i * dt
        if fire_at is not None and now >= fire_at:
            s._do_moment(flavor)
            fire_at = None
        if s.state == "armed" and s.swap_at is not None \
                and s.submix.clock >= s.swap_at:
            s._finish_swap()             # the tick loop's armed-state duty
        block = s.submix.read(BLOCK)
        # The DSP baseline a quiet mic path would produce; the coupler's
        # ground-truth branches must dominate all of it.
        state = {"beat": False, "beat_decay": 0.0, "beat_confidence": 0.3,
                 "beat_intensity": 0.0, "bass_punch": 0.0,
                 "audio_punch": 0.0, "bpm": 0.0, "beat_phase": 0.0,
                 "bar_phase": 0.0, "phrase_phase": 0.0,
                 "audio_energy": 0.4, "build_level": 0.0,
                 "drop": False, "drop_decay": 0.0}
        state.update(s.outstate_keys())
        try:
            lb = s.live_beat()
        except Exception:
            lb = None
        coupler.apply(state, dt, lb)
        tr["t"].append(now)
        tr["audio"].append(float(np.sqrt(np.mean(block ** 2))))
        tr["build"].append(state["build_level"])
        tr["drop_decay"].append(state["drop_decay"])
        tr["drop"].append(bool(state["drop"]))
        tr["beat_decay"].append(state["beat_decay"])
        tr["beat_intensity"].append(state["beat_intensity"])
        tr["hole"].append(bool(state.get("dj_moment_hole")))
        tr["eta"].append(state.get("dj_moment_eta"))
    return s, {k: np.array(v) if k != "eta" else v for k, v in tr.items()}


def half_life(t, decay, t0):
    """Seconds for drop_decay to fall from its stamp below 0.5."""
    idx = np.where((t >= t0) & (decay >= 0.99))[0]
    if not len(idx):
        return None
    for j in range(idx[0], len(t)):
        if decay[j] < 0.5:
            return t[j] - t[idx[0]]
    return None


def main():
    png = sys.argv[1] if len(sys.argv) > 1 else None

    print("\nmoment run (nextdrop - the one gesture)")
    s, m = run(24.0, fire_at=4.5)
    hit = s._moment_clock / RATE
    t = m["t"]
    pre = (t > 1.0) & (t < 4.0)
    hole = m["hole"]
    hole_span = t[hole]
    check("the grid-pulse floor beats the room before the press",
          float(m["beat_intensity"][pre].max()) > 0.5,
          f"max beat_intensity {m['beat_intensity'][pre].max():.2f}")
    lastq = (t > hit - 0.5) & (t < hit - 0.06)
    check("the build ramps to max into the landing",
          float(m["build"][lastq].min()) >= 0.9,
          f"build_level {m['build'][lastq].min():.2f} in the last half-second")
    check("the breath-hold spans the hole",
          len(hole_span) and abs(hole_span[0] - (hit - 0.5)) < 0.15
          and abs(hole_span[-1] - hit) < 0.15,
          f"hole {hole_span[0]:.2f}..{hole_span[-1]:.2f}s vs "
          f"{hit - 0.5:.2f}..{hit:.2f}s" if len(hole_span) else "no hole")
    check("beat pulses are suppressed through it",
          float(m["beat_intensity"][hole].max()) <= 0.1
          and float(m["beat_decay"][hole].max()) <= 0.15
          and float(m["build"][hole].min()) >= 1.0,
          f"intensity {m['beat_intensity'][hole].max():.2f}, "
          f"decay {m['beat_decay'][hole].max():.2f}, "
          f"build {m['build'][hole].min():.2f}")
    stamps = t[m["drop"]]
    check("the landing stamps the drop ON the hit",
          len(stamps) and abs(stamps[0] - hit) < 0.15,
          f"drop at {stamps[0]:.2f}s vs hit {hit:.2f}s"
          if len(stamps) else "never stamped")
    hl_hard = half_life(t, m["drop_decay"], stamps[0] if len(stamps) else hit)
    check("the hard slam sustains (tau 1.1)",
          hl_hard is not None and hl_hard > 0.55,
          f"decay half-life {hl_hard:.2f}s" if hl_hard else "no envelope")
    etas = [(tt, e) for tt, e in zip(t, m["eta"]) if e is not None]
    diffs = [b[1] - a[1] for a, b in zip(etas, etas[1:])]
    check("the published ETA counts down monotonically",
          len(etas) > 10 and max(diffs) < 1e-6 and abs(etas[-1][1]) < 0.3,
          f"{len(etas)} samples, worst step {max(diffs):+.3f}s"
          if diffs else "no eta published")

    print("\nnatural drop run (playhead crosses the drop section)")
    s2, m2 = run(12.0, cue_to=DROPX - 6.0)
    t2 = m2["t"]
    stamps2 = t2[m2["drop"]]
    check("a musical drop still stamps",
          len(stamps2) and abs(stamps2[0] - 6.0) < 0.3,
          f"drop at {stamps2[0]:.2f}s (crossing at ~6.0s)"
          if len(stamps2) else "never stamped")
    hl_nat = half_life(t2, m2["drop_decay"], stamps2[0]) \
        if len(stamps2) else None
    check("but flashes short (tau 0.35)",
          hl_nat is not None and hl_nat < 0.45,
          f"decay half-life {hl_nat:.2f}s" if hl_nat else "no envelope")
    check("the engineered slam is >2x the natural flash",
          hl_hard is not None and hl_nat is not None
          and hl_hard > 2.0 * hl_nat,
          f"hard {hl_hard}s vs natural {hl_nat}s")

    if png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 5))
            au = m["audio"] / max(m["audio"].max(), 1e-9)
            ax.fill_between(t, 0, au, color="0.85", label="audio (norm RMS)")
            ax.plot(t, m["build"], color="#e67e22", lw=2,
                    label="build_level")
            ax.plot(t, m["drop_decay"], color="#c0392b", lw=2,
                    label="drop_decay")
            ax.plot(t, m["beat_intensity"], color="#2980b9", lw=1,
                    label="beat_intensity")
            for a, b in [(hole_span[0], hole_span[-1])] if len(hole_span) \
                    else []:
                ax.axvspan(a, b, color="#2c3e50", alpha=0.25,
                           label="breath-hold (hole)")
            ax.axvline(4.5, color="0.4", ls=":", label="press")
            ax.axvline(hit, color="#c0392b", ls="--", label="landing")
            ax.set_xlabel("seconds")
            ax.set_ylim(0, 1.15)
            ax.legend(loc="upper left", ncol=3, fontsize=8)
            ax.set_title("Operator MOMENT (drop): what the shaders see")
            fig.tight_layout()
            fig.savefig(png, dpi=120)
            print(f"\nwrote {png}")
        except Exception as e:
            print(f"\nplot skipped: {e}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("ALL PASS")
    return 0


DROPX = T.DROP_S

if __name__ == "__main__":
    sys.exit(main())
