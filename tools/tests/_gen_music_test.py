"""Gate for the MUSIC of the generative system - the things that make a
set feel written rather than looped (docs/GENERATIVE_MUSIC_PLAN.md,
listening session 2):

  1. THEME: the lead makes a theme in a build and restates it on the
     downbeat of the drop; a key change retires it.
  2. INTERLOCK: bass hits avoid the kick's steps (downbeat excepted);
     keys stabs stay out of the lead's steps.
  3. COLOUR: borrowed chords and suspensions appear, and every pitched
     note is either in the key or in its bar's chord.
  4. TRANSITIONS: a riser runs into every drop, an impact lands on it.
  5. GROOVE: per-slot feel moves hats and snare in opposite directions.
  6. RACK: auto gain staging is deterministic and bounded, the same
     voice class under two patches renders two instruments, a kick
     chokes the previous kick, a sample slot plays a file, every voice
     class renders finite audio, and bus processing keeps the peak < 1.

Usage: python tools/tests/_gen_music_test.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.gen import RATE                                  # noqa: E402
from lib.gen.composer import Composer                      # noqa: E402
from lib.gen.composer.styles import get_style              # noqa: E402
from lib.gen.events import NoteEvent, DRUM_SLOTS           # noqa: E402
from lib.gen.synth import SynthRack                        # noqa: E402
from lib.gen.synth.voices import VOICES                    # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def phrases(style="groove", seed=11, minutes=12, bpm=124):
    c = Composer(style, bpm=bpm, key="8A", seed=seed)
    return c, list(c.phrases_until(int(minutes * 60 * RATE)))


def main():
    print("== theme")
    c, ps = phrases()
    ops = [(p.section, p.meta.get("lead_op")) for p in ps]
    made = [i for i, (s, o) in enumerate(ops) if o == "theme_make"]
    stated = [i for i, (s, o) in enumerate(ops) if o == "theme"]
    check(bool(made) and bool(stated), f"theme made in a build ({len(made)}x) and stated ({len(stated)}x)")
    check(all(ps[i].section == "drop" for i in stated), "every statement is on a drop")
    check(all(any(j < i for j in made) for i in stated), "each statement follows a making")
    # the statement reproduces the theme's rhythm on the drop's first phrase
    if stated:
        p = ps[stated[0]]
        lead = [e for e in p.events if e.slot == "lead"]
        check(len(lead) >= 3, f"stated theme has {len(lead)} lead notes")
    c2 = Composer("groove", bpm=124, key="8A", seed=11)
    for _ in range(4):
        c2.next_phrase()
    c2.melody.theme = object()
    c2.set_key("9A")
    c2.next_phrase()
    check(c2.melody.theme is None, "a key change retires the theme")

    print("== interlock")
    bad = tot = 0
    kb = tot_k = 0
    for p in ps:
        S = c.style["steps_per_bar"]
        spb = (p.end - p.start) / p.nbars
        sps = spb / S
        kick_steps = {}
        for e in p.events:
            if e.slot == "kick":
                b = int((e.at - p.start) // spb)
                kick_steps.setdefault(b, set()).add(int(round(((e.at - p.start) - b * spb) / sps)))
        lead_steps = {}
        for e in p.events:
            if e.slot == "lead":
                b = int((e.at - p.start) // spb)
                lead_steps.setdefault(b, set()).add(int(round(((e.at - p.start) - b * spb) / sps)))
        for e in p.events:
            b = int((e.at - p.start) // spb)
            st = int(round(((e.at - p.start) - b * spb) / sps))
            if e.slot == "bass" and st != 0:
                tot += 1
                bad += st in kick_steps.get(b, set())
            if e.slot == "keys" and lead_steps.get(b):
                tot_k += 1
                kb += st in lead_steps[b]
    check(tot > 100 and bad <= 0.02 * tot, f"bass off the kick: {bad}/{tot} collisions")
    check(tot_k > 20 and kb <= 0.05 * tot_k, f"keys out of the lead's steps: {kb}/{tot_k} collisions")

    print("== colour")
    borrowed = sum(1 for p in ps for ch in p.chords if ch[2].get("borrowed"))
    sus = sum(1 for p in ps for ch in p.chords if ch[2].get("sus"))
    pedal = sum(1 for p in ps for ch in p.chords if ch[2].get("pedal"))
    check(borrowed > 5 and sus > 5, f"borrowed chords {borrowed}, suspensions {sus}, pedal bars {pedal}")
    out = 0
    for p in ps:
        spb = (p.end - p.start) / p.nbars
        pcs = {c.key.degree_pc(d) for d in range(7)}
        allowed = [pcs | {m % 12 for m in c.harmony.notes(ch, 3, 4)} for ch in p.chords]
        for e in p.events:
            if e.slot in DRUM_SLOTS or e.slot in ("fx", "auto"):
                continue
            b = min(p.nbars - 1, max(0, int(round((e.at - p.start) / (spb / 16)) // 16)))   # nearest grid step's bar
            out += int(round(e.pitch)) % 12 not in allowed[b]
    check(out == 0, f"pitched notes in key or in their chord ({out} out)")

    print("== transitions")
    risers = [e for p in ps for e in p.events if e.slot == "fx" and e.params.get("kind") == "riser"]
    impacts = [e for p in ps for e in p.events if e.slot == "fx" and e.params.get("kind") == "impact"]
    hist = [h for h in c.form.history if h[0] < c.bar]                            # sections that were actually composed
    n_drops = sum(1 for _, sec, _ in hist if sec == "drop")                       # every drop section (re-drops too)
    n_built = sum(1 for i in range(1, len(hist)) if hist[i][1] == "drop" and hist[i - 1][1] == "build")
    check(n_built > 0 and len(risers) == n_built, f"{len(risers)} risers for {n_built} built-up drops")
    check(len(impacts) == n_drops, f"{len(impacts)} impacts for {n_drops} drop sections")
    ok_end = all(any(abs((r.at + r.dur) - ps[i].start) < 2 * RATE * 60 / 124 / 4 for i in range(1, len(ps))
                     if ps[i].section == "drop" and ps[i - 1].section == "build") for r in risers)
    check(ok_end, "every riser ends at its drop")

    print("== groove")
    c3 = Composer("groove", bpm=124, key="8A", seed=5)
    c3.form.section = "drop"; c3.form.bars_left = 10 ** 6; c3.form.hold = True
    p = c3.next_phrase()
    S = c3.style["steps_per_bar"]
    sps = (p.end - p.start) / p.nbars / S
    def offsets(slot):
        o = []
        for e in p.events:
            if e.slot == slot:
                st = round((e.at - p.start) / sps)
                o.append((e.at - p.start) - st * sps)
        return np.array(o) / RATE * 1000.0
    h, s = offsets("hat"), offsets("snare")
    check(len(h) > 10 and h.mean() > 2.0, f"hats sit late (mean {h.mean():+.1f} ms)")
    check(len(s) > 2 and s.mean() < 0.0, f"snare sits early (mean {s.mean():+.1f} ms)")
    c4 = Composer("groove", bpm=124, key="8A", seed=5)
    c4.form.section = "drop"; c4.form.bars_left = 10 ** 6; c4.form.hold = True
    c4.humanize = 0.0
    p4 = c4.next_phrase()
    sps4 = (p4.end - p4.start) / p4.nbars / S
    dev = max(abs((e.at - p4.start) - round((e.at - p4.start) / sps4) * sps4) for e in p4.events if e.slot in ("kick", "snare", "hat"))
    max_swing = c4.swing * max([1.0] + [float(f.get("swing", 1.0)) for f in c4.feel.values()]) * sps4
    check(dev <= 1.0 + max_swing, f"humanize 0 -> grid + swing only (max deviation {dev:.0f} samples, swing {max_swing:.0f})")

    print("== rack")
    st = get_style("groove")
    r1 = SynthRack(st, 124.0, seed=1)
    r2 = SynthRack(get_style("groove"), 124.0, seed=2)
    diff = {k: (round(r1.trim.get(k, -1), 4), round(r2.trim.get(k, -1), 4)) for k in set(r1.trim) | set(r2.trim)
            if abs(r1.trim.get(k, -1) - r2.trim.get(k, -1)) > 1e-3 * max(1.0, r1.trim.get(k, 1))}
    check(not diff and all(0.2 <= v <= 8.0 for v in r1.trim.values()) and len(r1.trim) >= 10,   # hosted slots may trim wider
          f"auto gain staging deterministic and bounded ({len(r1.trim)} slots, kick trim {r1.trim.get('kick', 0):.2f}{', differs: ' + str(diff) if diff else ''})")
    rng = np.random.default_rng(7)
    pad_g = VOICES["pad"]().render(57.0, 0.8, int(1.0 * RATE), get_style("groove")["slots"]["pad"], {}, rng)
    pad_a = VOICES["pad"]().render(57.0, 0.8, int(1.0 * RATE), get_style("ambient")["slots"]["pad"], {}, rng)
    check(pad_g.shape != pad_a.shape or not np.allclose(pad_g[: RATE // 2], pad_a[: RATE // 2]),
          "the same voice class under two style patches is two instruments")
    r = SynthRack(get_style("groove"), 124.0, seed=1)
    r.warm_up()
    gap = int(0.242 * RATE)
    r.schedule([NoteEvent(1000, "kick", 36.0, 1.0, 100, {}), NoteEvent(1000 + gap, "kick", 36.0, 1.0, 100, {})])
    while r.clock < 1000 + gap + 2048:
        r.render(2048)
    first_alive = any(it[0] == 1000 for it in r._active)
    second_alive = any(it[0] == 1000 + gap for it in r._active)
    check(second_alive and not first_alive, "a kick chokes the previous kick (its tail is gone once the next one starts)")
    import soundfile as sf
    tmp = os.path.join(tempfile.gettempdir(), "gen_sample_test.wav")
    t = np.arange(int(0.3 * RATE)) / RATE
    sf.write(tmp, (np.sin(2 * np.pi * 440 * t) * np.exp(-t / 0.1)).astype(np.float32), RATE)
    smp = VOICES["sample"]().render(72.0, 0.8, 100, {"file": tmp, "base_midi": 60}, {}, rng)
    check(smp.ndim == 2 and smp.shape[0] < int(0.3 * RATE) and np.abs(smp).max() > 0.1, "sample slot plays a file, pitched up = shorter")
    bad_voices = []
    for name, vc in VOICES.items():
        kinds = ["riser", "revcym", "impact", "sweep"] if name == "fx" else [None]
        for k in kinds:
            buf = vc().render(50.0, 0.8, int(0.4 * RATE), {"file": tmp} if name == "sample" else {}, {"kind": k} if k else {}, rng)
            if not (np.isfinite(buf).all() and np.abs(buf).max() > 1e-4):
                bad_voices.append(name + (f"/{k}" if k else ""))
    check(not bad_voices, f"every voice class renders finite, audible audio ({len(VOICES)} classes){' bad: ' + ','.join(bad_voices) if bad_voices else ''}")
    c5 = Composer("groove", bpm=124, key="8A", seed=9)
    c5.form.section = "drop"; c5.form.bars_left = 10 ** 6; c5.form.hold = True
    rk = SynthRack(c5.style, 124.0, seed=9)
    rk.warm_up()
    for p in c5.phrases_until(int(12 * RATE)):
        rk.schedule(p.events)
    blocks = []
    while rk.clock < 12 * RATE:
        blocks.append(rk.render(2048))
    x = np.concatenate(blocks)
    pk = float(np.abs(x).max())
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1])
    check(np.isfinite(x).all() and pk < 1.0, f"full drop through the bus chain: peak {pk:.3f}")
    check(corr < 0.97, f"stereo image (L/R correlation {corr:.3f})")
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
