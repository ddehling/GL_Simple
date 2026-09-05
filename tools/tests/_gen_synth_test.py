"""Gate for the generative SYNTH RACK (lib/gen/synth): the sound.

  1. DETERMINISM: same seed -> bit-identical render.
  2. CLEAN: finite, peak < 1.0 (soft clip), not silent, both channels.
  3. KICKS LAND ON THE GRID: in a kick-only render, low-band energy jumps
     at every scheduled kick (composer clock == rack clock).
  4. BUDGET: steady-state render at >= 3x realtime on this machine
     (numba warm) - the show box has ~1.2 s of ring to absorb spikes.
  5. TRACK PROTOCOL: read()/done/fade_out behave like an engine track.
  6. FLUIDSYNTH (if libfluidsynth + a SoundFont are present): SoundFont
     slots render audio and mix with the analog slots; else SKIP.

Usage: python tools/tests/_gen_synth_test.py [out.wav]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.synth import SynthRack                        # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def render(style="groove", seconds=20, seed=3, bpm=124, fluid=None, slots=(), only=None):
    c = Composer(style, bpm=bpm, key="8A", seed=seed)
    if only:
        c.muted = set(c.style["slots"]) - set(only)
    rack = SynthRack(c.style, c.bpm, fluid=fluid, fluid_slots=slots, seed=seed)
    kicks = []
    for p in c.phrases_until(seconds * RATE):
        rack.schedule(p.events)
        kicks += [e.at for e in p.events if e.slot == "kick"]
    out = []
    t0 = time.time()
    while rack.clock < seconds * RATE:
        out.append(rack.render(2048))
    return np.concatenate(out)[: seconds * RATE], kicks, time.time() - t0, rack


def kick_alignment(mix, kicks):
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, [30, 250], btype="band", fs=RATE, output="sos")
    low = sosfiltfilt(sos, mix.mean(axis=1))
    env = np.abs(low)
    w = int(0.025 * RATE)
    ok = 0
    for k in kicks:
        if k < w or k + w >= len(env):
            continue
        before = env[k - w:k - 2].mean() + 1e-6
        after = env[k + 2:k + w].mean()
        if after / before > 2.5:
            ok += 1
    return ok, len([k for k in kicks if w <= k < len(env) - w])


def main():
    print("== warm-up (numba JIT)")
    render(seconds=4)
    print("== determinism")
    a, kicks, _, _ = render()
    b, _, _, _ = render()
    check(np.array_equal(a, b), "same seed -> identical render")
    print("== clean")
    check(np.isfinite(a).all(), "finite")
    pk = float(np.abs(a).max())
    check(pk < 1.0, f"peak {pk:.3f} < 1.0")
    rms = float(np.sqrt(np.mean(a ** 2)))
    check(rms > 0.02, f"not silent (rms {20 * np.log10(rms):.1f} dBFS)")
    check(abs(a[:, 0]).sum() > 0 and abs(a[:, 1]).sum() > 0 and not np.array_equal(a[:, 0], a[:, 1]), "stereo")
    print("== kicks on the grid")
    # kick-only render: the sustained sub bass would mask the onset test,
    # and what is under test is the composer-clock -> rack-clock contract
    solo, kicks, _, _ = render(only=("kick",))
    hit, total = kick_alignment(solo, kicks)
    check(total > 20 and hit >= 0.95 * total, f"{hit}/{total} scheduled kicks produce a low-band onset")
    print("== budget")
    _, _, dt, rack = render(seconds=30)
    x = 30 / dt
    check(x >= 3.0, f"steady-state {x:.1f}x realtime ({rack.stats['notes']} notes)")
    print("== track protocol")
    c = Composer("groove", bpm=124, seed=1)
    r = SynthRack(c.style, c.bpm, seed=1)
    r.schedule(c.next_phrase().events)
    blk = r.read(1024)
    check(blk is not None and blk.shape == (1024, 2) and blk.dtype == np.float32, "read(n) -> (n,2) float32")
    check(r.is_narrative is False and r.is_ambient is False and r.is_soundpool is False, "engine flags present")
    r.fade_out(0.1)
    for _ in range(10):
        r.read(1024)
    check(r.done and r.read(1024) is None, "fade_out -> done -> read() returns None")
    print("== fluidsynth")
    try:
        from lib.gen.synth.fluid import FluidVoice
        fl = FluidVoice()
    except Exception as e:  # noqa: BLE001
        print(f"  SKIP fluidsynth unavailable ({e.__class__.__name__}: {e})")
        fl = None
    if fl is not None:
        m, _, _, rk = render(style="downtempo", seconds=12, seed=7, bpm=94, fluid=fl, slots=("keys", "pad"))
        check(np.isfinite(m).all() and float(np.abs(m).max()) < 1.0, "fluid+analog render clean")
        m2, _, _, _ = render(style="downtempo", seconds=12, seed=7, bpm=94)
        check(not np.array_equal(m, m2), "SoundFont slots change the sound")
    if len(sys.argv) > 1:
        import soundfile as sf
        sf.write(sys.argv[1], np.clip(a, -1, 1), RATE, subtype="PCM_16")
        print(f"wrote {sys.argv[1]}")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
