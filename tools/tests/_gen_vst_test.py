"""Gate for hosted instruments (lib/gen/synth/plugins.py): real VST3
instruments and effects inside the rack.

  1. SKIP cleanly when pedalboard or the plugin manifest is missing (a
     show never depends on a binary being present).
  2. A manifest entry resolves; the instrument renders a chord; render
     speed is reported.
  3. The style overlay turns a slot into a "vst" voice when the plugin
     exists and leaves it alone when it does not; the fallback patch is
     kept.
  4. A rack with a hosted slot renders the notes as one batch per phrase
     (the note-on lands on the grid, tails ring past note-off), through
     calibration, clean, and pre-rendered on the scheduling thread (the
     audio thread stays under budget).
  5. Muting a hosted slot silences it; a missing plugin falls back to
     the analog voice.
  6. A hosted effect on a bus processes audio with state across blocks.

Usage: python tools/tests/_gen_vst_test.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.composer.styles import get_style              # noqa: E402
from lib.gen.events import NoteEvent                       # noqa: E402
from lib.gen.synth import SynthRack, plugins               # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    if not plugins.available():
        print("SKIP: pedalboard not installed (pip install -r requirements-gen.txt)")
        return 0
    names = plugins.names("instrument")
    if not names:
        print("SKIP: no instruments in a plugins manifest (tools/gen/plugins.py scan media/plugins)")
        return 0
    name = "dexed" if "dexed" in names else names[0]
    print(f"== instrument vst:{name}")
    inst = plugins.instrument({"plugin": f"vst:{name}"})
    check(inst is not None, "manifest entry loads")
    if inst is None:
        return 1
    evs = [NoteEvent(0, "keys", 60.0, 0.9, RATE, {}), NoteEvent(0, "keys", 64.0, 0.9, RATE, {}), NoteEvent(RATE, "keys", 67.0, 0.8, RATE, {})]
    t0 = time.perf_counter()
    a = inst.render(evs, 0, 3.0)
    dt = time.perf_counter() - t0
    check(a.shape == (int(3.0 * RATE), 2) and np.isfinite(a).all() and np.abs(a).max() > 0.01,
          f"renders a chord ({a.shape[0]} smp, peak {np.abs(a).max():.3f}, {3 / dt:.0f}x realtime)")
    b = inst.render(evs, 0, 3.0)
    print(f"  info deterministic across calls: {np.array_equal(a, b)}")

    print("== overlay")
    st = get_style("groove")
    st["vst"] = {"keys": {"plugin": f"vst:{name}", "gain": 0.3, "tail": 1.0},
                 "arp": {"plugin": "vst:no_such_plugin", "gain": 0.3}}
    ov = plugins.overlay(st)
    check(ov["slots"]["keys"]["voice"] == "vst" and ov["slots"]["keys"]["fallback"]["voice"] == st["slots"]["keys"]["voice"],
          "slot becomes a vst voice with its analog patch kept as fallback")
    check(ov["slots"]["arp"]["voice"] == st["slots"]["arp"]["voice"], "a missing plugin leaves the slot alone")
    check(st["slots"]["keys"]["voice"] != "vst", "the original style is not modified")

    print("== rack")
    c = Composer("groove", bpm=124, key="8A", seed=31)
    c.form.section = "drop"; c.form.bars_left = 10 ** 6; c.form.hold = True
    c.muted = set(c.style["slots"]) - {"keys"}
    style = dict(c.style)
    style["vst"] = {"keys": {"plugin": f"vst:{name}", "gain": 0.4, "tail": 1.0}}
    r = SynthRack(style, 124.0, seed=31)
    r.warm_up()
    check(r.slots["keys"]["voice"] == "vst" and "keys" in r.trim, f"hosted slot calibrated (trim {r.trim.get('keys', 0):.2f})")
    keys_notes = []
    t0 = time.perf_counter()
    for p in c.phrases_until(int(12 * RATE)):
        r.schedule(p.events)
        keys_notes += [e for e in p.events if e.slot == "keys"]
    sched = time.perf_counter() - t0
    times = []
    out = []
    while r.clock < 12 * RATE:
        t1 = time.perf_counter()
        out.append(r.render(1024))
        times.append(time.perf_counter() - t1)
    x = np.concatenate(out)
    check(keys_notes and np.isfinite(x).all() and np.abs(x).max() > 0.01 and np.abs(x).max() <= 0.96,
          f"phrases render through the hosted slot ({len(keys_notes)} notes, peak {np.abs(x).max():.3f})")
    check(max(times) * 1000 < 8.0, f"audio thread stays light (max {max(times) * 1000:.1f} ms per 23 ms block; scheduling took {sched:.2f} s)")
    # the first note-on lands on the grid: energy appears at its sample, not before
    first = min(e.at for e in keys_notes)
    env = np.abs(x.mean(axis=1))
    w = int(0.01 * RATE)
    before = env[max(0, first - w):max(1, first - 2)].mean() + 1e-9
    after = env[first + 2:first + 4 * w].mean()
    check(after > 3 * before, f"first note lands on the grid (after/before {after / before:.1f})")
    check(r.stats.get("render_errors", 0) == 0, "no render errors")

    print("== mute + fallback")
    r2 = SynthRack(style, 124.0, seed=31)
    r2.set_mute("keys", True)
    c2 = Composer("groove", bpm=124, key="8A", seed=31)
    c2.form.section = "drop"; c2.form.bars_left = 10 ** 6; c2.form.hold = True
    c2.muted = set(c2.style["slots"]) - {"keys"}
    for p in c2.phrases_until(int(6 * RATE)):
        r2.schedule(p.events)
    y = np.concatenate([r2.render(2048) for _ in range(int(6 * RATE / 2048))])
    check(np.abs(y).max() < 1e-4, "a muted hosted slot is silent")
    style3 = dict(get_style("groove"))
    style3["slots"] = dict(style3["slots"])
    style3["slots"]["keys"] = dict(style3["slots"]["keys"], voice="vst", plugin="vst:no_such_plugin",
                                   fallback=dict(get_style("groove")["slots"]["keys"]))
    r3 = SynthRack(style3, 124.0, seed=31)
    c3 = Composer("groove", bpm=124, key="8A", seed=31)
    c3.form.section = "drop"; c3.form.bars_left = 10 ** 6; c3.form.hold = True
    c3.muted = set(c3.style["slots"]) - {"keys"}
    for p in c3.phrases_until(int(6 * RATE)):
        r3.schedule(p.events)
    z = np.concatenate([r3.render(2048) for _ in range(int(6 * RATE / 2048))])
    check(np.abs(z).max() > 0.01 and r3.stats.get("render_errors", 0) == 0, "a missing plugin falls back to the analog voice")

    print("== bus effect")
    fxn = plugins.names("effect")
    if not fxn:
        print("  SKIP no effect plugins in the manifest")
    else:
        fxp = plugins.effect({"plugin": f"vst:{fxn[0]}", "mix": 1.0})
        blk = np.random.default_rng(1).standard_normal((2048, 2)).astype(np.float32) * 0.1
        y1 = fxp.process(blk); y2 = fxp.process(blk)
        check(y1.shape == blk.shape and np.isfinite(y1).all(), f"vst:{fxn[0]} processes a block")
        print(f"  info state carries across blocks: {not np.array_equal(y1, y2)}")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
