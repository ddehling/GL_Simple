"""Gate for the Strudel composer path (lib/gen/composer/strudel.py +
tools/gen/strudel/bridge.mjs). SKIPs (exit 0) without node or before
`npm install` in tools/gen/strudel.

  1. BRIDGE: ping, eval, query; bad code returns an error, not a dead
     process; requests are answered in order.
  2. MAPPING: bd/cp/hh -> kick/snare/hat on the composer's sample grid;
     note names and numbers -> MIDI; gain -> velocity; lpf -> cutoff;
     unknown sounds dropped; drums ignore pitch.
  3. CONTEXT: a pattern reading `energy` follows the composer's form.
  4. IN THE SYSTEM: GenSystem.set_pattern replaces the notes from the next
     phrase, bad code is reported and survivable, clear_pattern returns to
     the rule composer; the rendered audio is clean and kicks hit the grid.

Usage: python tools/tests/_gen_strudel_test.py
"""
import collections
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.composer.strudel import (StrudelBridge, StrudelSource, StrudelV8,  # noqa: E402
                                      available, node_available, note_to_midi,
                                      open_engine, v8_available)
from lib.gen.system import GenSystem                      # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


CODE = ('stack(s("bd*4, [~ cp]*2, hh(5,8)"), note("<0 3 5 7>(3,8)").scale("A1:minor").s("bass").lpf(800),'
        ' note("c3 e3 g3 b3").s("lead").gain(energy), s("zzz*2"), note("60 62").s("pad"))')


def main():
    ok, why = available()
    if not ok:
        print(f"SKIP: {why}")
        return 0
    print("== engines")
    engines = []
    if v8_available()[0]:
        engines.append(("v8 (in-process)", StrudelV8()))
    else:
        print(f"  skip v8: {v8_available()[1]}")
    if node_available()[0]:
        engines.append(("node bridge", StrudelBridge()))
    else:
        print(f"  skip node: {node_available()[1]}")
    for name, eng in engines:
        check(eng.start(), f"{name}: ping")
        check(eng.eval(CODE), f"{name}: eval")
        haps = eng.query(0, 1, {"energy": 0.5})
        check(len(haps) > 10 and all("b" in h and "v" in h for h in haps), f"{name}: query -> {len(haps)} haps")
        g1 = {h["v"].get("gain") for h in eng.query(0, 1, {"energy": 0.9}) if h["v"].get("s") == "lead"}
        check(g1 == {0.9}, f"{name}: context signal read at query time {g1}")
        try:
            eng.eval("stack(("); check(False, f"{name}: bad code raises")
        except ValueError as e:
            check("token" in str(e).lower(), f"{name}: bad code -> error ({e})")
        check(eng.alive and eng.query(0, 1)[:1], f"{name}: survives bad code, old pattern intact")
        eng.stop()
    same = None
    if len(engines) == 2:
        a, bnode = StrudelV8(), StrudelBridge()
        a.start(); bnode.start(); a.eval(CODE); bnode.eval(CODE)
        same = a.query(0, 4, {"energy": 0.5}) == bnode.query(0, 4, {"energy": 0.5})
        check(same, "both engines produce identical events")
        a.stop(); bnode.stop()
    b = open_engine()
    print(f"  using {type(b).__name__} for the mapping checks")
    check(b.alive, "open_engine")
    check(b.eval(CODE), "eval")
    try:
        b.eval("stack(("); check(False, "bad code raises")
    except ValueError as e:
        check("Unexpected" in str(e) or "token" in str(e).lower(), f"bad code -> error ({e})")
    check(b.alive, "process survives bad code")
    haps = b.query(0, 1, {"energy": 0.5})
    check(len(haps) > 10 and all("b" in h and "v" in h for h in haps), f"query -> {len(haps)} haps")

    print("== mapping")
    check(note_to_midi("c3") == 48 and note_to_midi("eb4") == 63 and note_to_midi(60) == 60 and note_to_midi("x") is None, "note names/numbers")
    c = Composer("groove", bpm=124, key="8A", seed=1)
    src = StrudelSource(b, c.style["slots"].keys())
    src.load(CODE)
    c.pattern_source = src
    p = c.next_phrase()
    cnt = collections.Counter(e.slot for e in p.events)
    check(cnt["kick"] == 16 and cnt["snare"] == 8 and cnt["hat"] == 20, f"drums mapped {dict(cnt)}")
    check(cnt["lead"] == 16 and cnt["pad"] == 8 and "zzz" not in cnt, "pitched slots mapped, unknown sound dropped")
    beat = c.samples_per_beat
    kicks = sorted(e.at for e in p.events if e.slot == "kick")
    check(all(abs(k - round(k / beat) * beat) <= 1 for k in kicks), "kicks on the beat grid to the sample")
    lead = [e for e in p.events if e.slot == "lead"]
    check([e.pitch for e in lead[:4]] == [48.0, 52.0, 55.0, 59.0], f"lead pitches {[e.pitch for e in lead[:4]]}")
    check(all(abs(e.vel - p.energy) < 1e-3 for e in lead), "gain(energy) -> velocity follows the composer energy")
    bass = [e for e in p.events if e.slot == "bass"]
    check(bass and bass[0].params.get("cutoff") == 800.0 and bass[0].pitch == 33.0, f"bass note/lpf -> pitch {bass[0].pitch} cutoff {bass[0].params}")
    pads = [e for e in p.events if e.slot == "pad"]
    check(sorted({e.pitch for e in pads}) == [60.0, 62.0], "numeric notes -> MIDI")
    check(all(e.pitch == 36.0 for e in p.events if e.slot == "kick"), "drums ignore pitch")
    p2 = c.next_phrase()
    check(p2.start == p.end and p2.events and p2.events[0].at >= p2.start, "second phrase continues the timeline")
    b.stop()

    print("== in the system")
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=4, threaded=False, log_dir=tempfile.mkdtemp())
    g.start()
    out = []

    def pump(sec):
        for _ in range(int(sec * RATE / 2048)):
            blk = g.rack.read(2048)
            if blk is None:
                return
            out.append(blk); g.step()
    pump(3)
    g.set_pattern('s("bd*4")')
    pump(14)
    s = g.status()
    check(s["pattern"] == 's("bd*4")' and not s["pattern_error"] and not s["error"], "pattern accepted")
    ph = [p for p in g._phrases if p.start >= g.rack.clock - 8 * RATE]
    slots = collections.Counter(e.slot for p in ph for e in p.events if e.slot != "auto")   # mix automation is not a note
    check(set(slots) == {"kick"}, f"pattern replaced the rule composer's notes: {dict(slots)}")
    g.set_pattern("nope(("); pump(2)
    check(g.active and g.status()["error"].startswith("pattern:") and g.status()["pattern"] == 's("bd*4")', "bad code reported; old pattern keeps playing")
    g.clear_pattern(); pump(10)
    ph = [p for p in g._phrases if p.start >= g.rack.clock - 6 * RATE]
    slots = collections.Counter(e.slot for p in ph for e in p.events)
    check(len(slots) >= 3 and not g.status()["pattern"], f"clear -> rule composer back: {sorted(slots)}")
    g.stop()
    mix = np.concatenate(out)
    check(np.isfinite(mix).all() and float(np.abs(mix).max()) < 1.0, "audio clean")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
