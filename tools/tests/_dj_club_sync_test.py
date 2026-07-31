"""Gate: is the club actually IN SYNC with what the DJ plays?

The audio pipeline renders ~400ms AHEAD of the speakers (render-ahead
ring), telemetry snapshots once per 46ms render block, and the visual
frame loop runs at ~40 fps with jitter. Every one of those is a place
where the lights can drift off the music. This simulates the REAL
pipeline end to end - real DJSubmix render, real ring-lead accounting,
real outstate_keys()/live_beat()/DJVisualCoupler per visual frame - and
measures, in milliseconds, when the shaders see a beat/drop vs when the
speakers play it:

  1. BEAT ALIGNMENT at 120/174 bpm and off-unity rate: the coupler's
     beat flag vs the true audible kick times. Gate: |mean| < 35ms,
     p95 < 70ms (a light 70ms off a kick still reads as on-beat;
     100ms+ reads as lag).
  2. PHASE TRACKING: continuous beat_phase error (the +/-23ms
     nearest-snapshot telemetry quantization shows up here).
  3. CONTROL RUN with lead compensation OFF: errors must collapse to
     ~-400ms (visuals firing a beat early) - proving the sim actually
     models the ring and the compensation is doing real work.
  4. DROP STAMP: an operator MOMENT's landing stamp vs the audible
     landing. Gate: within 60ms.
  5. RESPONSIVENESS: beat flag -> pulse floor same frame; drop stamp ->
     full drop_decay same frame; energy step follows the deliberate
     0.8s smoothing (reported, not gated - it's a design choice).

Usage:
    python tools/tests/_dj_club_sync_test.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _dj_moment_test as T                       # noqa: E402
import _dj_moment_vis_test as V                   # noqa: E402
from lib.dj.submix import RATE                    # noqa: E402
from lib.dj.vis import DJVisualCoupler            # noqa: E402

FPS = 40.0                       # project default target_fps
LEAD_S = 0.400                   # audio_engine RING_TARGET_MS
BLOCK = 2048                     # audio_engine RENDER_BLOCK (~46ms)
failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


class RingEngine:
    """The render-ahead ring's timing contract, nothing else."""

    def __init__(self):
        self.produced = 0        # frames rendered into the ring
        self.heard = 0.0         # frames the speakers have played

    def render_lead_frames(self):
        return max(0, int(self.produced - self.heard))


def simulate(seconds, bpm=120.0, rate=1.0, fire_at=None, comp=True,
             energy_step=None, seed=7):
    """Run the visual frame loop against a speaker-time ground truth.

    Returns per-frame traces keyed by HEARD time. `comp=False` runs the
    control: engine detached, so _render_lead()=0 and the visuals read
    render-head state raw (the pre-ring bug)."""
    T.Track.period_s = 60.0 / bpm
    s = V.vis_system()
    if rate != 1.0:
        s.submix.post({"cmd": "rate", "deck": "a", "value": rate})
    eng = RingEngine()
    s.engine = eng if comp else None
    coupler = DJVisualCoupler()
    rng = np.random.RandomState(seed)
    lead = int(LEAD_S * RATE)
    tr = {k: [] for k in ("t", "beat", "phase", "drop", "decay",
                          "pulse", "energy")}
    heard_t = 0.0
    while heard_t < seconds:
        dt = 1.0 / FPS + rng.uniform(-0.006, 0.006)   # frame jitter
        heard_t += dt
        eng.heard = heard_t * RATE
        # The producer keeps the ring topped up to the target lead.
        while eng.produced - eng.heard < lead:
            s.submix.read(BLOCK)
            eng.produced += BLOCK
        if fire_at is not None and heard_t >= fire_at:
            s._do_moment("drop")
            fire_at = None
        if energy_step is not None:
            s.live_energy = (lambda v: (lambda: v))(
                0.9 if heard_t >= energy_step else 0.3)
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
        tr["t"].append(heard_t)
        tr["beat"].append(bool(state["beat"]))
        tr["phase"].append(float(state["beat_phase"]))
        tr["drop"].append(bool(state["drop"]))
        tr["decay"].append(float(state["drop_decay"]))
        tr["pulse"].append(float(state["bass_punch"]))
        tr["energy"].append(float(state["audio_energy"]))
    T.Track.period_s = 60.0 / T.BPM
    return s, {k: np.array(v) for k, v in tr.items()}


CUE_S = 40.0                     # make_system's load cue


def beat_errors(tr, bpm, rate):
    """Signed ms between each beat flag and the nearest TRUE audible
    beat. Truth from the SOURCE beat grid: the deck cues to CUE_S -
    which need not sit on a beat line (at 128 bpm it does not; assuming
    it did put a 150ms lie in this gate's first version) - and plays at
    `rate`, so source beat k*per_src hits the speakers at
    (k*per_src - CUE_S) / rate."""
    per_src = 60.0 / bpm
    per = per_src / rate
    off = (-CUE_S / rate) % per          # heard time of a source beat, mod per
    flags = tr["t"][tr["beat"]]
    errs = [((f - off + per / 2) % per) - per / 2 for f in flags]
    return np.array(errs) * 1000.0, flags


def phase_error(tr, bpm, rate):
    """Circular RMS error (beats) of the continuous phase signal."""
    per_src = 60.0 / bpm
    true = ((CUE_S + rate * tr["t"]) / per_src) % 1.0
    err = ((tr["phase"] - true + 0.5) % 1.0) - 0.5
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    print("\n1-2. beat alignment + phase tracking (lead compensated)")
    for bpm, rate in [(120.0, 1.0), (174.0, 1.0), (128.0, 1.03)]:
        _s, tr = simulate(20.0, bpm=bpm, rate=rate)
        errs, flags = beat_errors(tr, bpm, rate)
        pe = phase_error(tr, bpm, rate)
        n_expect = 20.0 / ((60.0 / bpm) / rate)
        ok = (len(flags) > 0.9 * n_expect
              and abs(float(np.mean(errs))) < 35.0
              and float(np.percentile(np.abs(errs), 95)) < 70.0
              and pe < 0.08)
        check(f"{bpm:.0f} bpm rate {rate}", ok,
              f"{len(flags)} beats, mean {np.mean(errs):+.0f}ms, "
              f"p95 {np.percentile(np.abs(errs), 95):.0f}ms, "
              f"worst {np.max(np.abs(errs)):.0f}ms, "
              f"phase RMS {pe:.3f} beats")

    print("\n3. control: compensation OFF (the pre-ring bug)")
    _s, tr0 = simulate(20.0, comp=False)
    errs0, _ = beat_errors(tr0, 120.0, 1.0)
    # Uncompensated state runs LEAD_S ahead of the speakers; folded into
    # +/-half-period the 400ms lead at 120bpm reads as ~-100ms mod 500ms.
    exp = ((-LEAD_S + 0.25) % 0.5 - 0.25) * 1000.0
    check("visuals fire early without the ring lead",
          abs(float(np.mean(errs0)) - exp) < 40.0,
          f"mean {np.mean(errs0):+.0f}ms (expected ~{exp:+.0f}ms folded: "
          f"the raw lead is {LEAD_S * 1000:.0f}ms)")

    print("\n4. drop stamp on the audible landing (operator MOMENT)")
    s, trm = simulate(24.0, fire_at=4.5)
    hit_heard = s._moment_clock / RATE   # vclock == heard clock domain
    stamps = trm["t"][trm["drop"]]
    err_ms = (stamps[0] - hit_heard) * 1000.0 if len(stamps) else None
    check("stamp lands with the audio",
          err_ms is not None and abs(err_ms) < 60.0,
          f"stamp {err_ms:+.0f}ms from the audible landing"
          if err_ms is not None else "never stamped")

    print("\n5. responsiveness")
    i = np.argmax(trm["decay"] >= 0.99)
    check("the drop flash is full-strength the same frame it stamps",
          trm["drop"][i], f"decay 0.99 at frame of stamp={bool(trm['drop'][i])}")
    _s, tre = simulate(12.0, energy_step=4.0)
    t63 = tre["t"][(tre["t"] > 4.0) & (tre["energy"] > 0.3 + 0.63 * 0.6)]
    check("energy follows a step with the designed ~0.8s smoothing",
          len(t63) and 0.4 < (t63[0] - 4.0) < 1.4,
          f"63% at +{t63[0] - 4.0:.2f}s" if len(t63) else "never rose")
    on = trm["pulse"][trm["beat"]]
    check("the pulse floor peaks on the beat frame itself",
          len(on) and float(np.min(on)) > 0.5,
          f"min bass_punch on beat frames {np.min(on):.2f}"
          if len(on) else "no beats")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
