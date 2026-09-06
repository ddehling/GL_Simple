"""Gate for the melody sources and development (listening session 4):
authored hooks, the corpus model, and the operators that turn a motif
into a phrase.

  1. MODEL: the corpus model loads (SKIP that block if it has not been
     built); motifs sampled from it move mostly by step, resolve leaps
     by step, and cadence on tonic / third / fifth.
  2. OPERATORS: sequence shifts the cell, augment slows it, fragment
     repeats the head, retrograde reverses the line.
  3. HOOKS: validate() rejects junk and accepts a good hook; a HookAuthor
     with a fake transport caches hooks and its provider hands them out;
     a theme made with a provider IS the hook (name, rhythm) and its
     answer fills bars 2-3.
  4. PHRASE: the last phrase of a section cadences on the tonic, held;
     the last phrase of a build climaxes an octave up.
  5. SYSTEM: with hooks disabled nothing is requested; status reports
     the melody source.

Usage: python tools/tests/_gen_melody_test.py
"""
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.composer import melody_model                  # noqa: E402
from lib.gen.composer.hooks import HookAuthor, validate    # noqa: E402
from lib.gen.composer.melody import develop                # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def hold(c, section, bars_left=10 ** 6):
    c.form.section = section
    c.form.bars_left = bars_left
    c.form.hold = bars_left > 100
    return c


def main():
    print("== corpus model")
    if not melody_model.available():
        print("  SKIP model not built (python tools/gen/melody_corpus.py)")
    else:
        rng = random.Random(3)
        steps_total = leaps = resolved = cad = n_motifs = 0
        for _ in range(300):
            steps = melody_model.sample_rhythm(rng, 7)
            degs = melody_model.sample_line(rng, steps, cadence=True)
            if not degs:
                continue
            n_motifs += 1
            ivs = [b - a for a, b in zip(degs, degs[1:])]
            steps_total += sum(1 for v in ivs if abs(v) <= 1)
            for i in range(1, len(ivs)):
                if abs(ivs[i - 1]) >= 3:
                    leaps += 1
                    resolved += ivs[i] * ivs[i - 1] < 0 and abs(ivs[i]) <= 2
            cad += degs[-1] % 7 in (0, 2, 4)
        tot_iv = sum(1 for _ in range(1)) and max(1, n_motifs * 6)
        step_share = steps_total / tot_iv
        check(n_motifs >= 250 and step_share >= 0.5, f"corpus lines move mostly by step ({step_share:.0%} of intervals)")
        check(leaps == 0 or resolved / leaps >= 0.7, f"leaps resolve by step ({resolved}/{leaps})")
        check(cad >= 0.95 * n_motifs, f"cadence on 1, 3 or 5 ({cad}/{n_motifs})")
        hist = melody_model.interval_hist()
        check(sum(v for k, v in hist.items() if abs(k) <= 1) > 0.5, "the corpus itself is mostly stepwise (sanity)")

    print("== operators")
    rng = random.Random(1)
    st, dg = [0, 4, 6, 10, 12], [0, 2, 3, 2, 0]
    s2, d2 = develop("sequence", st, dg, rng)
    check(len(s2) == 10 and s2[5:] == [s + 16 for s in st] and all(b - a == d2[5] - dg[0] for a, b in zip(dg, d2[5:])),
          "sequence repeats the cell a step away in bar 2")
    s3, d3 = develop("augment", st, dg, rng)
    check(s3 == [0, 8, 12, 20, 24] and d3 == dg, "augment doubles the durations")
    s4, d4 = develop("fragment", st, dg, rng)
    check(s4[:3] == [0, 4, 6] and s4[3:6] == [8, 12, 14] and len(s4) >= 9, "fragment repeats the head")
    s5, d5 = develop("retrograde", st, dg, rng)
    check(d5 == dg[::-1] and s5 == st, "retrograde reverses the line on the same rhythm")

    print("== hooks")
    check(validate({"steps": [0, 3, 6, 10, 12, 15, 19, 22, 28], "degrees": [0, 2, 4, 7, 4, 2, 0, -3, 0], "contour": "arch"}) is not None,
          "a good hook validates")
    check(validate({"steps": [0, 1], "degrees": [0, 1]}) is None and validate({"steps": "x"}) is None
          and validate({"steps": [1, 3, 5, 7, 9], "degrees": [0, 0, 0, 0, 0]}) is None,
          "junk hooks are rejected (too short, malformed, never on a beat)")
    tmp = os.path.join(tempfile.gettempdir(), "gen_hooks_test.json")
    if os.path.exists(tmp):
        os.remove(tmp)
    reply = ('{"hooks": [{"name": "riff A", "steps": [0, 3, 6, 10, 12, 16, 22, 28], "degrees": [0, 2, 4, 2, 0, 4, 2, 0], "contour": "arch",'
             ' "answer": {"steps": [0, 4, 8, 12, 16, 24], "degrees": [4, 2, 0, -1, 2, 0]}},'
             ' {"name": "bad", "steps": [1], "degrees": [0]}]}')
    ha = HookAuthor(tmp, transport=lambda sys_p, p: reply, enabled=True)
    ha.request("groove", "club groove", 124.0, "A minor", n=2, block=True)
    check(ha.count("groove") == 1 and not ha.error, f"the author caches the valid hook only ({ha.count('groove')} cached, error {ha.error!r})")
    ha2 = HookAuthor(tmp, enabled=False)
    check(ha2.count("groove") == 1, "hooks persist on disk for the next night")
    prov = ha.provider("groove")
    h = prov(random.Random(0))
    check(h and h["name"] == "riff A" and "answer" in h, "provider hands out the hook with its answer")
    c = Composer("groove", bpm=124, key="8A", seed=4)
    c.melody.hook_provider = prov
    hold(c, "build", bars_left=8)
    p = c.next_phrase()
    S = 16
    sps = (p.end - p.start) / p.nbars / S
    lead = sorted((round((e.at - p.start) / sps), e) for e in p.events if e.slot == "lead")
    steps = [s for s, _ in lead]
    check(p.meta.get("lead_op") == "theme_make" and c.melody.theme is not None and c.melody.theme.name == "riff A"
          and c.melody.source == "hook", "the theme made in the build is the authored hook")
    bar01 = [s for s in steps if s < 2 * S]
    bar23 = [s - 2 * S for s in steps if s >= 2 * S]
    check(set(bar01) >= {0, 3, 6, 10, 12, 16, 22, 28} - {28} and all(s in (0, 4, 8, 12, 16, 24) for s in bar23) and len(bar23) >= 4,
          f"bars 0-1 play the hook, bars 2-3 its answer ({bar01} | {bar23})")

    print("== phrase")
    c2 = Composer("groove", bpm=124, key="8A", seed=6)
    hold(c2, "drop", bars_left=4)             # this phrase ends the section
    p2 = c2.next_phrase()
    lead2 = sorted(((e.at, e) for e in p2.events if e.slot == "lead"))
    last = lead2[-1][1] if lead2 else None
    check(last is not None and int(round(last.pitch)) % 12 == c2.key.degree_pc(0) and last.dur >= 2.5 * (p2.end - p2.start) / 64,
          f"a section's last phrase cadences on the tonic, held ({last.dur / RATE if last else 0:.2f} s)")
    c3 = Composer("groove", bpm=124, key="8A", seed=6)
    hold(c3, "build", bars_left=8)
    p3a = c3.next_phrase()
    p3b = c3.next_phrase()                     # last phrase of the build: climax
    def mean_pitch(p):
        ps = [e.pitch for e in p.events if e.slot == "lead"]
        return sum(ps) / len(ps) if ps else 0
    check(mean_pitch(p3b) >= mean_pitch(p3a) + 6, f"the build's last phrase climaxes higher ({mean_pitch(p3a):.1f} -> {mean_pitch(p3b):.1f})")

    print("== system")
    os.environ["GEN_HOOKS"] = "0"
    from lib.gen.system import GenSystem
    g = GenSystem(engine=None, style="groove", bpm=124, key="8A", seed=5, set_length_s=1800, log_dir="logs", threaded=False)
    g.start()
    for _ in range(20):
        g.rack.read(4096); g.step()
    st = g.status()
    check(not g.hooks.enabled and st["hooks"]["pending"] == 0 and st["hooks"]["source"] in ("hook", "corpus", "walk"),
          f"hooks off in gates; melody source reported: {st['hooks']['source']}")
    g.stop()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
