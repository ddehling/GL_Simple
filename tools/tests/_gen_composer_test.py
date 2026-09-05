"""Gate for the generative COMPOSER (lib/gen/composer): the notes, before
any synth touches them.

  1. DETERMINISM: same style/seed/key/bpm -> identical events.
  2. TIMELINE: phrases are contiguous, events sorted, inside their phrase.
  3. IN KEY: every pitched note is a scale tone of the current key.
  4. FORM MOVES: over 10 minutes every style leaves its first section and
     (club styles) reaches a drop; the drop is announced in phrase meta
     before it lands (the visuals count down to it).
  5. MOTIF MEMORY: lead phrases reuse material (an op other than 'new').
  6. STEERING: a key change requested mid-run lands at a phrase boundary.

Usage: python tools/tests/_gen_composer_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer, STYLES              # noqa: E402
from lib.gen.events import DRUM_SLOTS                      # noqa: E402
from lib.gen.theory import parse_key                       # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def run(style, seed=11, minutes=10):
    c = Composer(style, seed=seed, key="8A")
    ps = list(c.phrases_until(int(minutes * 60 * RATE)))
    return c, ps


def main():
    print("== determinism")
    a = [(e.at, e.slot, e.pitch, e.vel, e.dur) for p in run("groove", 3, 3)[1] for e in p.events]
    b = [(e.at, e.slot, e.pitch, e.vel, e.dur) for p in run("groove", 3, 3)[1] for e in p.events]
    check(a == b and len(a) > 200, f"same seed -> identical {len(a)} events")
    c2 = [(e.at, e.slot, e.pitch) for p in run("groove", 4, 3)[1] for e in p.events]
    check(c2 != a, "different seed -> different events")

    for style in STYLES:
        print(f"== {style}")
        c, ps = run(style)
        ok = all(q.start == p.end for p, q in zip(ps, ps[1:]))
        check(ok, f"{len(ps)} phrases contiguous")
        ok = all(all(p.events[i].at <= p.events[i + 1].at for i in range(len(p.events) - 1)) for p in ps)
        check(ok, "events sorted")
        ok = all(p.start <= e.at < p.end for p in ps for e in p.events)
        check(ok, "events inside their phrase")
        bad = 0
        for p in ps:
            key = parse_key(p.meta["camelot"]) if False else None
            k = c.key  # key is constant in this run
            pcs = {k.degree_pc(d) for d in range(7)}
            for e in p.events:
                if e.slot in DRUM_SLOTS:
                    continue
                if int(round(e.pitch)) % 12 not in pcs:
                    bad += 1
        check(bad == 0, f"all pitched notes in {c.key.name} ({bad} out)")
        sections = [s for _, s, _ in c.form.history]
        check(len(set(sections)) >= 3, f"form moves: {sections[:8]}")
        if style != "ambient":
            check("drop" in sections, "reaches a drop")
            announced = [p for p in ps if p.drops()]
            landed = [p for p in ps if p.section == "drop"]
            check(bool(announced) and announced[0].bar0 < landed[0].bar0, "drop announced before it lands")
            ops = {p.meta["lead_op"] for p in ps if p.meta["lead_op"]}
            check(bool(ops - {"new"}), f"motif memory reused: {sorted(ops)}")
        n = sum(len(p.events) for p in ps)
        check(n > 500, f"{n} notes in 10 min")

    print("== steering")
    c = Composer("groove", seed=5, key="8A")
    p1 = c.next_phrase()
    c.set_key("9A")
    p2 = c.next_phrase()
    check(p1.key.startswith("A") and p2.key.startswith("E"), f"key change at phrase boundary: {p1.key} -> {p2.key}")
    c.set_bpm(128.0)
    p3 = c.next_phrase()
    check(p3.bpm == 128.0 and p3.start == p2.end, "tempo change at phrase boundary keeps the timeline contiguous")

    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
